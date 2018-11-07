import numpy as np
import matplotlib.pyplot as plt
from crnn_pytorch.CRNN import img_read


# Here the priors with different aspect ratios are calculated and passed while encoding
#and decoding function


class prior_util:

    def __init__(self,model):

        textbox_layers_names=[l.name.split('/')[0] for l in model.textbox_layers]
        self.textbox_layers=textbox_layers_names
        self.model=model
        self.image_size=model.input_shape[1:3]
        num_maps=len(textbox_layers_names)

        aspect_ratios=model.aspect_ratios
        shifts=model.shifts
        steps=model.steps

        min_dim=np.min(self.image_size)
        min_ratio=10
        max_ratio=100
        s=np.linspace(min_ratio,max_ratio,num_maps+1) * min_dim/100
        minmax_sizes=[(round(s[i]),round(s[i+1])) for i in range(len(s)-1)]

        minmax_sizes=np.array(minmax_sizes) * 1

        #clips
        #special ssd assignment

        self.ssd_assignment=True

        self.prior_maps=[]
        for i in range(num_maps):
            layer=model.get_layer(textbox_layers_names[i])
            map_size=layer.output_shape[1:3]
            m=prior_map(textbox_layer_name=textbox_layers_names[i],
                         image_size=self.image_size,
                         map_size=map_size,
                         minmax_size=minmax_sizes[i],
                         variances=[0.1, 0.1, 0.2, 0.2],
                         aspect_ratios=aspect_ratios[i],
                         shift=shifts[i],
                         steps=steps[i])
            self.prior_maps.append(m)

        self.update_priors()


        self.nms_top_k = 400
        self.nms_thresh = 0.45


    def update_priors(self):
        priors_xy = []
        priors_wh = []
        priors_min_xy = []
        priors_max_xy = []
        priors_variances = []
        priors = []

        map_offsets = [0]
        for i in range(len(self.prior_maps)):
            m = self.prior_maps[i]

            # collect prior data
            priors_xy.append(m.priors_xy)
            priors_wh.append(m.priors_wh)
            priors_min_xy.append(m.priors_min_xy)
            priors_max_xy.append(m.priors_max_xy)
            priors_variances.append(m.priors_variances)
            priors.append(m.priors)
            map_offsets.append(map_offsets[-1] + len(m.priors))

        self.priors_xy = np.concatenate(priors_xy, axis=0)
        self.priors_wh = np.concatenate(priors_wh, axis=0)
        self.priors_min_xy = np.concatenate(priors_min_xy, axis=0)
        self.priors_max_xy = np.concatenate(priors_max_xy, axis=0)
        self.priors_variances = np.concatenate(priors_variances, axis=0)
        self.priors = np.concatenate(priors, axis=0)
        self.map_offsets = map_offsets

        # normalized prior boxes
        image_h, image_w = self.image_size
        self.priors_xy_norm = self.priors_xy / (image_w, image_h)
        self.priors_wh_norm = self.priors_wh / (image_w, image_h)
        self.priors_min_xy_norm = self.priors_min_xy / (image_w, image_h)
        self.priors_max_xy_norm = self.priors_max_xy / (image_w, image_h)
        self.priors_norm = np.concatenate([self.priors_min_xy_norm, self.priors_max_xy_norm, self.priors_variances],
                                          axis=1)

    def encode(self,gt_data,overlap_threshold=0.5):


        num_classes = self.model.num_classes
        num_priors = self.priors.shape[0]

        gt_boxes = self.gt_boxes = np.copy(gt_data[:, :4])  # normalized xmin, ymin, xmax, ymax
        gt_class_idx = np.asarray(gt_data[:, -1] + 0.5, dtype=np.int)
        gt_one_hot = np.zeros([len(gt_class_idx), num_classes])
        gt_one_hot[range(len(gt_one_hot)), gt_class_idx] = 1

        gt_min_xy = gt_boxes[:, 0:2]
        gt_max_xy = gt_boxes[:, 2:4]
        gt_xy = (gt_boxes[:, 2:4] + gt_boxes[:, 0:2]) / 2.
        gt_wh = gt_boxes[:, 2:4] - gt_boxes[:, 0:2]

        gt_iou = np.array([self.iou(b, self.priors_norm) for b in gt_boxes]).T
        max_idxs = np.argmax(gt_iou, axis=1)

        priors_xy = self.priors_xy_norm
        priors_wh = self.priors_wh_norm

        max_idxs = np.argmax(gt_iou, axis=1)
        max_val = gt_iou[np.arange(num_priors), max_idxs]
        prior_mask = max_val > overlap_threshold
        match_indices = max_idxs[prior_mask]

        self.match_indices = dict(zip(list(np.ix_(prior_mask)[0]), list(match_indices)))

        # confidence
        confidence = np.zeros((num_priors, num_classes))
        confidence[:, 0] = 1
        confidence[prior_mask] = gt_one_hot[match_indices]

        # local offsets
        gt_xy = gt_xy[match_indices]
        gt_wh = gt_wh[match_indices]
        priors_xy = priors_xy[prior_mask]
        priors_wh = priors_wh[prior_mask]
        variances_xy = self.priors[prior_mask, -4:-2]
        variances_wh = self.priors[prior_mask, -2:]
        offsets = np.zeros((num_priors, 4))
        offsets[prior_mask, 0:2] = (gt_xy - priors_xy) / priors_wh
        offsets[prior_mask, 2:4] = np.log(gt_wh / priors_wh)
        offsets[prior_mask, 0:2] /= variances_xy
        offsets[prior_mask, 2:4] /= variances_wh

        return np.concatenate([offsets, confidence], axis=1)

    def iou(self,box, priors):

        inter_upleft = np.maximum(priors[:, :2], box[:2])
        inter_botright = np.minimum(priors[:, 2:4], box[2:])
        inter_wh = inter_botright - inter_upleft
        inter_wh = np.maximum(inter_wh, 0)
        inter = inter_wh[:, 0] * inter_wh[:, 1]
        # compute union
        area_pred = (box[2] - box[0]) * (box[3] - box[1])
        area_gt = (priors[:, 2] - priors[:, 0]) * (priors[:, 3] - priors[:, 1])
        union = area_pred + area_gt - inter

        # compute iou
        iou = inter / union
        return iou

    def decode(self, model_output, confidence_threshold=0.01, keep_top_k=200, sparse=True):
        # calculation is done with normalized sizes

        prior_mask = model_output[:, 4:] > confidence_threshold

        if sparse:
            # compute boxes only if the confidence is high enough and the class is not background
            mask = np.any(prior_mask[:, 1:], axis=1)
            prior_mask = prior_mask[mask]
            mask = np.ix_(mask)[0]
            model_output = model_output[mask]
            priors_xy = self.priors_xy[mask] / self.image_size
            priors_wh = self.priors_wh[mask] / self.image_size
            priors_variances = self.priors[mask, -4:]
        else:
            priors_xy = self.priors_xy / self.image_size
            priors_wh = self.priors_wh / self.image_size
            priors_variances = self.priors[:, -4:]

        offsets = model_output[:, :4]
        confidence = model_output[:, 4:]

        num_priors = offsets.shape[0]
        num_classes = confidence.shape[1]

        # compute bounding boxes from local offsets
        boxes = np.empty((num_priors, 4))
        offsets = offsets * priors_variances
        boxes_xy = priors_xy + offsets[:, 0:2] * priors_wh
        boxes_wh = priors_wh * np.exp(offsets[:, 2:4])
        boxes[:, 0:2] = boxes_xy - boxes_wh / 2.  # xmin, ymin
        boxes[:, 2:4] = boxes_xy + boxes_wh / 2.  # xmax, ymax
        boxes = np.clip(boxes, 0.0, 1.0)

        # do non maximum suppression
        results = []
        for c in range(1, num_classes):
            mask = prior_mask[:, c]
            boxes_to_process = boxes[mask]
            if len(boxes_to_process) > 0:
                confs_to_process = confidence[mask, c]


                idx = non_maximum_suppression(
                    boxes_to_process, confs_to_process,
                    self.nms_thresh, self.nms_top_k)


                good_boxes = boxes_to_process[idx]
                good_confs = confs_to_process[idx][:, None]
                labels = np.ones((len(idx), 1)) * c
                c_pred = np.concatenate((good_boxes, good_confs, labels), axis=1)
                results.extend(c_pred)
        if len(results) > 0:
            results = np.array(results)
            order = np.argsort(-results[:, 4])
            results = results[order]
            results = results[:keep_top_k]
        else:
            results = np.empty((0, 6))
        self.results = results
        return results

    def plot_results(self, results=None, classes=None, show_labels=True, gt_data=None, confidence_threshold=None,image=None):
        if results is None:
            results = self.results
        if confidence_threshold is not None:
            mask = results[:, 4] > confidence_threshold
            results = results[mask]
        if classes is not None:
            colors = plt.cm.hsv(np.linspace(0, 1, len(classes) + 1)).tolist()
        ax = plt.gca()
        im = plt.gci()
        image_size = im.get_size()

        # draw ground truth
        if gt_data is not None:
            for box in gt_data:
                label = np.nonzero(box[4:])[0][0] + 1
                color = 'g' if classes == None else colors[label]
                xy_rec = to_rec(box[:4], image_size)
                ax.add_patch(plt.Polygon(xy_rec, fill=True, color=color, linewidth=1, alpha=0.3))

        # draw prediction
        for r in results:
            label = int(r[5])
            confidence = r[4]
            color = 'r' if classes == None else colors[label]
            xy_rec = to_rec(r[:4], image_size)
            ax.add_patch(plt.Polygon(xy_rec, fill=False, edgecolor=color, linewidth=2))
            if show_labels:
                if image is not None:
                    text=img_read(crop_img(r[0:4],image))
                label_name = label if classes == None else text
                xmin, ymin = xy_rec[0]
                display_txt = '%0.2f, %s' % (confidence, label_name)
                ax.text(xmin, ymin, display_txt, bbox={'facecolor': color, 'alpha': 0.5})


def to_rec(box, image_size):

    img_height, img_width = image_size
    xmin = np.min(box[0::2]) * img_width
    xmax = np.max(box[0::2]) * img_width
    ymin = np.min(box[1::2]) * img_height
    ymax = np.max(box[1::2]) * img_height
    xy_rec = np.array([[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]])
    return xy_rec


def to_rec2(box, image_size):

    img_height, img_width = image_size
    xmin = np.min(box[0::2]) * img_width
    xmax = np.max(box[0::2]) * img_width
    ymin = np.min(box[1::2]) * img_height
    ymax = np.max(box[1::2]) * img_height
    xy_rec = np.array([xmin,xmax,ymin,ymax])
    return xy_rec



def crop_img(box,img):
    t = to_rec2(box,img.shape[0:2])
    t = list(map(int,t))
    roi = img[t[2]:t[3],t[0]:t[1]]
    return roi



def non_maximum_suppression(boxes, confs, overlap_threshold, top_k):

    eps = 1e-15

    boxes = boxes.astype(np.float64)

    pick = []
    x1, y1, x2, y2 = boxes.T

    idxs = np.argsort(confs)
    area = (x2 - x1) * (y2 - y1)

    while len(idxs) > 0:
        i = idxs[-1]

        pick.append(i)
        if len(pick) >= top_k:
            break

        idxs = idxs[:-1]

        xx1 = np.maximum(x1[i], x1[idxs])
        yy1 = np.maximum(y1[i], y1[idxs])
        xx2 = np.minimum(x2[i], x2[idxs])
        yy2 = np.minimum(y2[i], y2[idxs])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        I = w * h

        overlap = I / (area[idxs] + eps)

        idxs = idxs[overlap <= overlap_threshold]

    return pick


class prior_map():

    def __init__(self,textbox_layer_name,image_size,map_size,minmax_size,variances,
                 aspect_ratios,shift,steps):

        self.__dict__.update(locals())
        self.compute_priors()


    def compute_priors(self):
        image_h, image_w = self.image_size
        map_h, map_w = self.map_size
        min_size, max_size = self.minmax_size

        step_x = step_y = self.steps
        linx = np.array([(0.5 + i) for i in range(map_w)]) * step_x
        liny = np.array([(0.5 + i) for i in range(map_h)]) * step_y

        box_xy = np.array(np.meshgrid(linx, liny)).reshape(2, -1).T

        shift=self.shift

        box_wh = []
        box_shift = []
        for i in range(len(self.aspect_ratios)):
            ar = self.aspect_ratios[i]
            box_wh.append([min_size * np.sqrt(ar), min_size / np.sqrt(ar)])
            box_shift.append(shift[i])

        box_wh = np.asarray(box_wh)

        box_shift = np.asarray(box_shift)
        box_shift = np.clip(box_shift, -1.0, 1.0)
        box_shift = box_shift * 0.5 * np.array([step_x, step_y])

        #individual priors

        priors_shift = np.tile(box_shift, (len(box_xy), 1))
        priors_xy = np.repeat(box_xy, len(box_wh), axis=0) + priors_shift
        priors_wh = np.tile(box_wh, (len(box_xy), 1))

        priors_min_xy = priors_xy - priors_wh / 2.
        priors_max_xy = priors_xy + priors_wh / 2.
        priors_variances = np.tile(self.variances, (len(priors_xy), 1))

        self.box_xy = box_xy
        self.box_wh = box_wh
        self.box_shift = box_shift

        self.priors_xy = priors_xy
        self.priors_wh = priors_wh
        self.priors_min_xy = priors_min_xy
        self.priors_max_xy = priors_max_xy
        self.priors_variances = priors_variances
        self.priors = np.concatenate([priors_min_xy, priors_max_xy, priors_variances], axis=1)






