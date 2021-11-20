'''
Implementation of Intersection over Union algorithm:

Suppose we have batches of boxes with shapes (N, 4)
where N is a batch size (number of boxes)
Take coordinates from batch
Note: I used slices because I want coordinates to have shape (N, 1)
if I would use just boxes_pred[..., 0] the shape will be (N) (the second dim will be removed)
'''

import torch


def intersection_over_union(boxes_pred: torch.tensor, boxes_gt: torch.tensor, box_format="midpoint") -> torch.tensor:
    '''
    Params:
        boxes_pred (tensor): prediction bounding boxes of shape (batch_size, 4)
        boxes_gt (tensor): ground truth (correct) bounding boxes of shape (batch_size, 4)
        box_format (str): corners/midpoint, if boxes (x1,y1,x2,y2) or (x,y,h,w)
    Return:
        tensor: intersection over union for all boxes in the batch
    '''

    if box_format == "midpoint":
        # in this case boxes are looks like (x,y, height, width) where x,y - middle of the box
        box1_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
        box1_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
        box1_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3] / 2
        box1_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4] / 2
        box2_x1 = boxes_gt[..., 0:1] - boxes_gt[..., 2:3] / 2
        box2_y1 = boxes_gt[..., 1:2] - boxes_gt[..., 3:4] / 2
        box2_x2 = boxes_gt[..., 2:3] + boxes_gt[..., 2:3] / 2
        box2_y2 = boxes_gt[..., 3:4] + boxes_gt[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_pred[..., 0:1]
        box1_y1 = boxes_pred[..., 1:2]
        box1_x2 = boxes_pred[..., 2:3]
        box1_y2 = boxes_pred[..., 3:4]
        box2_x1 = boxes_gt[..., 0:1]
        box2_y1 = boxes_gt[..., 1:2]
        box2_x2 = boxes_gt[..., 2:3]
        box2_y2 = boxes_gt[..., 3:4]

    # coordinates for finding intersection area
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # clamp(0) is for the case if boxes don't intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection

    return intersection / union


if __name__ == '__main__':
    box1 = torch.tensor([2, 2, 6, 6])
    box2 = torch.tensor([4, 4, 7, 8])

    correct1 = 1 / 6

    print("IoU1 = ", intersection_over_union(box1, box2, box_format="corners"))
    assert torch.eq(intersection_over_union(box1, box2, box_format="corners"), correct1), "doesnt work correctly"

    boxes_pred = torch.tensor(
        [
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 2, 2],
            [0, 0, 3, 2],
        ]
    )
    boxes_gt = torch.tensor(
        [
            [3, 0, 5, 2],
            [3, 0, 5, 2],
            [0, 3, 2, 5],
            [2, 0, 5, 2],
            [1, 1, 3, 3],
            [1, 1, 3, 3],
        ]
    )
    correct2 = torch.tensor([0, 0, 0, 0, 1 / 7, 0.25])
    print("IoU2 = ", intersection_over_union(boxes_pred, boxes_gt, box_format="corners"))
    assert sum(torch.abs(intersection_over_union(boxes_pred, boxes_gt, box_format="corners") -
                         correct2.unsqueeze(1))) < 0.001, "doesnt work correctly"
