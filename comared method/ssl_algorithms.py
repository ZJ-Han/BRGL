import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch

def masked_average(losses, weights):
    weights = weights.float()
    denom = weights.sum().clamp_min(1.0)
    return (losses * weights).sum() / denom

def hard_cross_entropy(logits, targets):
    return F.cross_entropy(logits, targets, reduction='none')

def soft_cross_entropy(logits, targets):
    return -(targets * F.log_softmax(logits, dim=1)).sum(dim=1)

def one_hot(targets, num_classes):
    return F.one_hot(targets, num_classes=num_classes).float()

def gaussian_similarity_loss(z1, z2, sigma=0.5, eps=1e-08):
    distance_sq = (z1 - z2).pow(2).sum(dim=1)
    similarity = torch.exp(-distance_sq / max(2.0 * sigma * sigma, eps))
    return -torch.log(similarity.clamp_min(eps)).mean()

class ModelEMA:

    def __init__(self, model, decay):
        self.model = copy.deepcopy(model)
        self.model.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model):
        ema_state = self.model.state_dict()
        model_state = model.state_dict()
        for key, value in model_state.items():
            if key not in ema_state:
                continue
            if not torch.is_floating_point(value):
                ema_state[key].copy_(value)
                continue
            ema_state[key].mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

class AdaptiveThresholdController:

    def __init__(self, num_classes, base_threshold=0.95, momentum=0.999):
        self.base_threshold = base_threshold
        self.momentum = momentum
        self.global_conf = base_threshold
        self.class_hist = torch.full((num_classes,), 1.0 / num_classes)

    @torch.no_grad()
    def thresholds(self, probs):
        device = probs.device
        self.class_hist = self.class_hist.to(device)
        max_probs, pseudo = probs.max(dim=1)
        batch_conf = max_probs.mean().item()
        batch_hist = probs.mean(dim=0)
        self.global_conf = self.momentum * self.global_conf + (1.0 - self.momentum) * batch_conf
        self.class_hist.mul_(self.momentum).add_(batch_hist.detach(), alpha=1.0 - self.momentum)
        normalized_hist = self.class_hist / self.class_hist.max().clamp_min(1e-06)
        global_threshold = torch.full_like(max_probs, self.global_conf)
        return global_threshold * normalized_hist[pseudo]

def _to_unit_range(x):
    return x.add(1.0).mul(0.5).clamp(0.0, 1.0)

def _from_unit_range(x):
    return x.mul(2.0).sub(1.0).clamp(-1.0, 1.0)

def random_horizontal_flip(x, p=0.5):
    if x.size(0) == 0:
        return x
    mask = torch.rand(x.size(0), device=x.device) < p
    out = x.clone()
    if mask.any():
        out[mask] = torch.flip(out[mask], dims=[3])
    return out

def random_crop(x, padding=None):
    batch_size, _, height, width = x.size()
    if padding is None:
        padding = max(4, int(round(0.125 * min(height, width))))
    x_pad = F.pad(x, (padding, padding, padding, padding), mode='reflect')
    max_y = x_pad.size(2) - height
    max_x = x_pad.size(3) - width
    ys = torch.randint(0, max_y + 1, (batch_size,), device=x.device)
    xs = torch.randint(0, max_x + 1, (batch_size,), device=x.device)
    out = torch.empty_like(x)
    for index in range(batch_size):
        out[index] = x_pad[index, :, ys[index]:ys[index] + height, xs[index]:xs[index] + width]
    return out

def weak_augment(x):
    return random_crop(random_horizontal_flip(x))

def _adjust_brightness(img, strength):
    img_unit = _to_unit_range(img)
    return _from_unit_range((img_unit * strength).clamp(0.0, 1.0))

def _adjust_contrast(img, strength):
    img_unit = _to_unit_range(img)
    mean = img_unit.mean(dim=(2, 3), keepdim=True)
    return _from_unit_range(((img_unit - mean) * strength + mean).clamp(0.0, 1.0))

def _adjust_saturation(img, strength):
    img_unit = _to_unit_range(img)
    gray = img_unit.mean(dim=1, keepdim=True)
    return _from_unit_range(((img_unit - gray) * strength + gray).clamp(0.0, 1.0))

def _adjust_gamma(img, strength):
    img_unit = _to_unit_range(img)
    return _from_unit_range(img_unit.pow(strength).clamp(0.0, 1.0))

def _solarize(img, threshold):
    img_unit = _to_unit_range(img)
    out = torch.where(img_unit < threshold, img_unit, 1.0 - img_unit)
    return _from_unit_range(out)

def _blur(img):
    padded = F.pad(img, (1, 1, 1, 1), mode='reflect')
    return F.avg_pool2d(padded, kernel_size=3, stride=1)

def _sharpen(img, strength):
    blurred = _blur(img)
    return (img + strength * (img - blurred)).clamp(-1.0, 1.0)

def _add_noise(img, strength):
    return (img + torch.randn_like(img) * strength).clamp(-1.0, 1.0)

def cutout_batch(x, ratio=0.5, fill=0.0):
    if ratio <= 0.0:
        return x
    out = x.clone()
    _, _, height, width = out.size()
    cut_h = max(1, int(height * ratio))
    cut_w = max(1, int(width * ratio))
    for index in range(out.size(0)):
        center_y = torch.randint(0, height, (1,), device=out.device).item()
        center_x = torch.randint(0, width, (1,), device=out.device).item()
        y1 = max(0, center_y - cut_h // 2)
        y2 = min(height, center_y + cut_h // 2)
        x1 = max(0, center_x - cut_w // 2)
        x2 = min(width, center_x + cut_w // 2)
        out[index, :, y1:y2, x1:x2] = fill
    return out

def strong_augment(x, num_ops=2, cutout_ratio=0.4):
    ops = (lambda img: _adjust_brightness(img, 0.6 + 0.8 * torch.rand(1, device=img.device).item()), lambda img: _adjust_contrast(img, 0.6 + 0.8 * torch.rand(1, device=img.device).item()), lambda img: _adjust_saturation(img, 0.6 + 0.8 * torch.rand(1, device=img.device).item()), lambda img: _adjust_gamma(img, 0.7 + 0.6 * torch.rand(1, device=img.device).item()), lambda img: _solarize(img, 0.35 + 0.3 * torch.rand(1, device=img.device).item()), lambda img: _sharpen(img, 0.3 + 0.7 * torch.rand(1, device=img.device).item()), lambda img: _add_noise(img, 0.02 + 0.04 * torch.rand(1, device=img.device).item()), _blur)
    out = weak_augment(x)
    if out.size(0) == 0:
        return out
    for index in range(out.size(0)):
        img = out[index:index + 1]
        op_order = torch.randperm(len(ops), device=out.device)[:num_ops].tolist()
        for op_index in op_order:
            img = ops[op_index](img)
        out[index] = img[0]
    return cutout_batch(out, ratio=cutout_ratio)

def cutmix_batch(x, alpha=1.0):
    batch_size = x.size(0)
    if batch_size < 2:
        return (x, torch.arange(batch_size, device=x.device), 1.0)
    beta_dist = torch.distributions.Beta(alpha, alpha)
    lam = float(beta_dist.sample().item())
    perm = torch.randperm(batch_size, device=x.device)
    _, _, height, width = x.size()
    cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
    cut_w = max(1, int(width * cut_ratio))
    cut_h = max(1, int(height * cut_ratio))
    center_x = torch.randint(0, width, (1,), device=x.device).item()
    center_y = torch.randint(0, height, (1,), device=x.device).item()
    x1 = max(0, center_x - cut_w // 2)
    x2 = min(width, center_x + cut_w // 2)
    y1 = max(0, center_y - cut_h // 2)
    y2 = min(height, center_y + cut_h // 2)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
    box_area = max(1, (x2 - x1) * (y2 - y1))
    lam_adjusted = 1.0 - float(box_area) / float(height * width)
    return (mixed, perm, lam_adjusted)

def train_one_epoch_fixmatch(model, ema, optimizer, sup_loader, unsup_loader, device, tau, lambda_u, ra_ops, cutout_ratio):
    model.train()
    running_sup = 0.0
    running_unsup = 0.0
    running_mask = 0.0
    num_batches = 0
    sup_iter = infinite_loader(sup_loader)
    for x_u, _, _, _ in unsup_loader:
        x_l, y_l, _, _ = next(sup_iter)
        x_l = x_l.to(device)
        y_l = y_l.to(device)
        x_u = x_u.to(device)
        x_l_w = weak_augment(x_l)
        logits_l = model(x_l_w)
        loss_sup = F.cross_entropy(logits_l, y_l)
        with torch.no_grad():
            logits_u_w = model(weak_augment(x_u))
            probs_u_w = F.softmax(logits_u_w, dim=1)
            conf, pseudo = probs_u_w.max(dim=1)
            mask = (conf >= tau).float()
        logits_u_s = model(strong_augment(x_u, num_ops=ra_ops, cutout_ratio=cutout_ratio))
        unsup_losses = hard_cross_entropy(logits_u_s, pseudo)
        loss_unsup = masked_average(unsup_losses, mask)
        loss = loss_sup + lambda_u * loss_unsup
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        running_sup += loss_sup.item()
        running_unsup += loss_unsup.item()
        running_mask += mask.mean().item()
        num_batches += 1
    return (running_sup / max(1, num_batches), running_unsup / max(1, num_batches), running_mask / max(1, num_batches))

def train_one_epoch_infomatch(model, ema, optimizer, sup_loader, unsup_loader, device, controller, tau, lambda_u, lambda_c, gaussian_sigma, cutmix_alpha, ra_ops, cutout_ratio, num_classes):
    model.train()
    running_sup = 0.0
    running_weak_strong = 0.0
    running_cutmix = 0.0
    running_contrast = 0.0
    running_mask = 0.0
    num_batches = 0
    sup_iter = infinite_loader(sup_loader)
    for x_u, _, _, _ in unsup_loader:
        x_l, y_l, _, _ = next(sup_iter)
        x_l = x_l.to(device)
        y_l = y_l.to(device)
        x_u = x_u.to(device)
        x_l_w = weak_augment(x_l)
        logits_l = model(x_l_w)
        loss_sup = F.cross_entropy(logits_l, y_l)
        with torch.no_grad():
            logits_u_w = model(weak_augment(x_u))
            probs_u_w = F.softmax(logits_u_w, dim=1)
            conf, pseudo = probs_u_w.max(dim=1)
            if controller is not None:
                thresholds = controller.thresholds(probs_u_w)
            else:
                thresholds = torch.full_like(conf, tau)
            mask = (conf >= thresholds).float()
            pseudo_targets = one_hot(pseudo, num_classes=num_classes)
        x_u_s = strong_augment(x_u, num_ops=ra_ops, cutout_ratio=cutout_ratio)
        x_u_mix, perm, lam = cutmix_batch(weak_augment(x_u), alpha=cutmix_alpha)
        logits_u_s, proj_u_s = model(x_u_s, return_projection=True)
        logits_u_mix, proj_u_mix = model(x_u_mix, return_projection=True)
        loss_weak_strong = masked_average(hard_cross_entropy(logits_u_s, pseudo), mask)
        mixed_targets = lam * pseudo_targets + (1.0 - lam) * pseudo_targets[perm]
        mixed_mask = lam * mask + (1.0 - lam) * mask[perm]
        loss_cutmix = masked_average(soft_cross_entropy(logits_u_mix, mixed_targets), mixed_mask)
        loss_contrast = gaussian_similarity_loss(proj_u_s, proj_u_mix, sigma=gaussian_sigma)
        loss = loss_sup + lambda_u * (loss_weak_strong + loss_cutmix) + lambda_c * loss_contrast
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        running_sup += loss_sup.item()
        running_weak_strong += loss_weak_strong.item()
        running_cutmix += loss_cutmix.item()
        running_contrast += loss_contrast.item()
        running_mask += mask.mean().item()
        num_batches += 1
    return (running_sup / max(1, num_batches), running_weak_strong / max(1, num_batches), running_cutmix / max(1, num_batches), running_contrast / max(1, num_batches), running_mask / max(1, num_batches))

def train_one_epoch_maxmatch(model, ema, optimizer, sup_loader, unsup_loader, device, lambda_u, tau, temperature, num_views, ra_ops, cutout_ratio):
    model.train()
    running_sup = 0.0
    running_unsup = 0.0
    running_mask = 0.0
    num_batches = 0
    sup_iter = infinite_loader(sup_loader)
    for x_u, _, _, _ in unsup_loader:
        x_l, y_l, _, _ = next(sup_iter)
        x_l = x_l.to(device)
        y_l = y_l.to(device)
        x_u = x_u.to(device)
        logits_l = model(weak_augment(x_l))
        loss_sup = F.cross_entropy(logits_l, y_l)
        with torch.no_grad():
            logits_anchor = model(x_u)
            anchor_probs = F.softmax(logits_anchor / max(temperature, 1e-06), dim=1).detach()
            if tau > 0.0:
                mask = (anchor_probs.max(dim=1).values >= tau).float()
            else:
                mask = torch.ones(anchor_probs.size(0), device=device)
        view_losses = []
        for _ in range(num_views):
            logits_view = model(strong_augment(x_u, num_ops=ra_ops, cutout_ratio=cutout_ratio))
            consistency = F.kl_div(F.log_softmax(logits_view, dim=1), anchor_probs, reduction='none').sum(dim=1)
            view_losses.append(consistency)
        worst_case_loss = torch.stack(view_losses, dim=0).max(dim=0).values
        loss_unsup = masked_average(worst_case_loss, mask)
        loss = loss_sup + lambda_u * loss_unsup
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if ema is not None:
            ema.update(model)
        running_sup += loss_sup.item()
        running_unsup += loss_unsup.item()
        running_mask += mask.mean().item()
        num_batches += 1
    return (running_sup / max(1, num_batches), running_unsup / max(1, num_batches), running_mask / max(1, num_batches))

def _check_logits(logits_uw, logits_us, num_classes):
    if logits_uw.ndim != 2 or logits_us.shape != logits_uw.shape:
        raise ValueError('weak/strong logits must have the same [batch, class] shape')
    if logits_uw.shape[0] == 0 or logits_uw.shape[1] != num_classes:
        raise ValueError('invalid unlabeled batch or class dimension')

class FlexState:

    def __init__(self, num_classes, pool_size, p_cutoff=0.95, thresh_warmup=True):
        if num_classes < 2 or pool_size < 1 or (not 0.0 < p_cutoff <= 1.0):
            raise ValueError('invalid FlexMatch configuration')
        self.num_classes = int(num_classes)
        self.pool_size = int(pool_size)
        self.p_cutoff = float(p_cutoff)
        self.thresh_warmup = bool(thresh_warmup)
        self.selected_label = torch.full((pool_size,), -1, dtype=torch.long)
        self.classwise_acc = torch.zeros(num_classes)
        self.steps = 0

    def _update_classwise(self):
        counts = torch.bincount(self.selected_label + 1, minlength=self.num_classes + 1)
        largest = int(counts.max().item())
        if largest < self.pool_size:
            denominator = largest if self.thresh_warmup else int(counts[1:].max().item())
            self.classwise_acc.copy_(counts[1:].to(self.classwise_acc.dtype) / denominator)

    def compute(self, logits_uw, logits_us, ulb_indices, logits_lb=None):
        _check_logits(logits_uw, logits_us, self.num_classes)
        if ulb_indices is None:
            raise ValueError('FlexMatch requires stable unlabeled-pool indices')
        indices = torch.as_tensor(ulb_indices, dtype=torch.long, device=logits_uw.device)
        if indices.ndim != 1 or len(indices) != len(logits_uw):
            raise ValueError('one pool index is required per unlabeled sample')
        if int(indices.min().item()) < 0 or int(indices.max().item()) >= self.pool_size:
            raise ValueError('unlabeled index outside the frozen pool')
        self.selected_label = self.selected_label.to(logits_uw.device)
        self.classwise_acc = self.classwise_acc.to(logits_uw.device)
        with torch.no_grad():
            self._update_classwise()
            probabilities = logits_uw.detach().softmax(dim=-1)
            confidence, pseudo = probabilities.max(dim=-1)
            classwise = self.classwise_acc[pseudo]
            thresholds = self.p_cutoff * classwise / (2.0 - classwise)
            weights = confidence.ge(thresholds).to(logits_us.dtype)
            selected = confidence.ge(self.p_cutoff)
            self.selected_label[indices[selected]] = pseudo[selected]
        unsup_loss = (F.cross_entropy(logits_us, pseudo, reduction='none') * weights).mean()
        self.steps += 1
        return (unsup_loss, {'weights': weights.detach(), 'pseudo_labels': pseudo.detach(), 'selection': selected.detach(), 'thresholds': thresholds.detach(), 'mask_mean': float(weights.mean().item()), 'confidence_mean': float(confidence.mean().item()), 'bank_assigned_fraction': float(self.selected_label.ge(0).float().mean().item()), 'classwise_acc': self.classwise_acc.detach().clone()})

    def state_dict(self):
        return {'algorithm': 'flexmatch', 'num_classes': self.num_classes, 'pool_size': self.pool_size, 'p_cutoff': self.p_cutoff, 'thresh_warmup': self.thresh_warmup, 'steps': self.steps, 'selected_label': self.selected_label.detach().cpu().clone(), 'classwise_acc': self.classwise_acc.detach().cpu().clone()}

    def load_state_dict(self, state):
        for key in ('num_classes', 'pool_size', 'p_cutoff', 'thresh_warmup'):
            if state[key] != getattr(self, key):
                raise ValueError('FlexMatch checkpoint configuration mismatch: ' + key)
        if state.get('algorithm') != 'flexmatch':
            raise ValueError('wrong algorithm state')
        bank = state['selected_label'].detach().cpu().clone()
        classwise = state['classwise_acc'].detach().cpu().clone()
        if bank.shape != (self.pool_size,) or classwise.shape != (self.num_classes,):
            raise ValueError('invalid FlexMatch state shapes')
        if bank.min() < -1 or bank.max() >= self.num_classes:
            raise ValueError('invalid pseudo-label bank')
        self.selected_label, self.classwise_acc = (bank.long(), classwise.float())
        self.steps = int(state['steps'])

class SoftState:

    def __init__(self, num_classes, ema_p=0.999, dist_align=True):
        if num_classes < 2 or not 0.0 <= ema_p < 1.0:
            raise ValueError('invalid SoftMatch configuration')
        self.num_classes = int(num_classes)
        self.ema_p = float(ema_p)
        self.dist_align = bool(dist_align)
        self.lb_prob_t = torch.ones(num_classes) / num_classes
        self.ulb_prob_t = torch.ones(num_classes) / num_classes
        self.prob_max_mu_t = 1.0 / num_classes
        self.prob_max_var_t = 1.0
        self.steps = 0

    @torch.no_grad()
    def _update_prob(self, lb_probs, ulb_probs):
        self.lb_prob_t = self.ema_p * self.lb_prob_t + (1.0 - self.ema_p) * lb_probs.mean(0)
        self.ulb_prob_t = self.ema_p * self.ulb_prob_t + (1.0 - self.ema_p) * ulb_probs.mean(0)
        max_probs = ulb_probs.max(dim=-1)[0]
        self.prob_max_mu_t = self.ema_p * self.prob_max_mu_t + (1.0 - self.ema_p) * max_probs.mean().item()
        self.prob_max_var_t = self.ema_p * self.prob_max_var_t + (1.0 - self.ema_p) * max_probs.var(unbiased=True).item()

    def compute(self, logits_uw, logits_us, ulb_indices=None, logits_lb=None):
        _check_logits(logits_uw, logits_us, self.num_classes)
        if len(logits_uw) < 2:
            raise ValueError('SoftMatch unbiased variance requires an unlabeled batch of at least two')
        if logits_lb is None or logits_lb.ndim != 2 or logits_lb.shape[1] != self.num_classes or (len(logits_lb) == 0):
            raise ValueError('SoftMatch requires labeled predictions for the official distribution alignment')
        self.lb_prob_t = self.lb_prob_t.to(logits_uw.device)
        self.ulb_prob_t = self.ulb_prob_t.to(logits_uw.device)
        with torch.no_grad():
            raw_probs = logits_uw.detach().softmax(dim=-1)
            lb_probs = logits_lb.detach().softmax(dim=-1)
            self._update_prob(lb_probs, raw_probs)
            aligned = raw_probs
            if self.dist_align:
                aligned = raw_probs * self.lb_prob_t / self.ulb_prob_t
                aligned = aligned / aligned.sum(dim=1, keepdim=True)
            confidence = aligned.max(dim=-1)[0]
            if not math.isfinite(self.prob_max_var_t) or self.prob_max_var_t <= 0.0:
                raise FloatingPointError('SoftMatch confidence variance is not finite and positive')
            weights = torch.exp(-torch.clamp(confidence - self.prob_max_mu_t, max=0.0).pow(2) / (2.0 * self.prob_max_var_t / 4.0))
            pseudo = raw_probs.argmax(dim=-1)
        unsup_loss = (F.cross_entropy(logits_us, pseudo, reduction='none') * weights).mean()
        self.steps += 1
        return (unsup_loss, {'weights': weights.detach(), 'pseudo_labels': pseudo.detach(), 'mask_mean': float(weights.mean().item()), 'confidence_mean': float(confidence.mean().item()), 'raw_confidence_mean': float(raw_probs.max(dim=-1)[0].mean().item()), 'mu': self.prob_max_mu_t, 'variance': self.prob_max_var_t, 'lb_prob_t': self.lb_prob_t.detach().clone(), 'ulb_prob_t': self.ulb_prob_t.detach().clone()})

    def state_dict(self):
        return {'algorithm': 'softmatch', 'num_classes': self.num_classes, 'ema_p': self.ema_p, 'dist_align': self.dist_align, 'steps': self.steps, 'lb_prob_t': self.lb_prob_t.detach().cpu().clone(), 'ulb_prob_t': self.ulb_prob_t.detach().cpu().clone(), 'prob_max_mu_t': self.prob_max_mu_t, 'prob_max_var_t': self.prob_max_var_t}

    def load_state_dict(self, state):
        for key in ('num_classes', 'ema_p', 'dist_align'):
            if state[key] != getattr(self, key):
                raise ValueError('SoftMatch checkpoint configuration mismatch: ' + key)
        if state.get('algorithm') != 'softmatch':
            raise ValueError('wrong algorithm state')
        for key in ('lb_prob_t', 'ulb_prob_t'):
            value = state[key].detach().cpu().clone().float()
            if value.shape != (self.num_classes,) or not bool(torch.isfinite(value).all()) or (not bool(value.gt(0).all())):
                raise ValueError('invalid SoftMatch distribution state')
            setattr(self, key, value)
        self.prob_max_mu_t = float(state['prob_max_mu_t'])
        self.prob_max_var_t = float(state['prob_max_var_t'])
        if not math.isfinite(self.prob_max_mu_t) or not math.isfinite(self.prob_max_var_t) or self.prob_max_var_t <= 0:
            raise ValueError('invalid SoftMatch Gaussian state')
        self.steps = int(state['steps'])

def call_bn(bn, x):
    return bn(x)

class Classifier2(nn.Module):

    def __init__(self, num_classes=3):
        super(Classifier2, self).__init__()
        self.dropout_rate = 0.2
        self.c1 = nn.Conv2d(3, 32, kernel_size=11, stride=4, padding=0)
        self.c2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=0)
        self.c3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.l_c1 = nn.Linear(25600, 1024)
        self.l_c2 = nn.Linear(1024, num_classes)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)

    def forward(self, x):
        h = x
        h = self.c1(h)
        h = F.leaky_relu(h, negative_slope=0.01)
        h = F.dropout2d(h, p=self.dropout_rate)
        h = self.c2(h)
        h = F.leaky_relu(call_bn(self.bn2, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h = F.dropout2d(h, p=self.dropout_rate)
        h = self.c3(h)
        h = F.leaky_relu(call_bn(self.bn3, h), negative_slope=0.01)
        h = F.max_pool2d(h, kernel_size=3, stride=1)
        h1 = h.view(h.size(0), -1)
        h2 = self.l_c1(h1)
        logits = self.l_c2(h2)
        return (logits, h2)
