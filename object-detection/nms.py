import torch
from IoU import intersection_over_union


def non_max_suppression(bboxes,
                        iou_threshold,
                        prob_threshold,
                        box_format="corners") -> list:

    # bboxes = [[class, probability, x1, y1, x2, y2], [class, ...], ...]
    # bboxes[...,0] = class
    # bboxes[...,1] = probability
    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > prob_threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        # updating list of bboxes
        bboxes = [box for box in bboxes
                  if box[0] != chosen_box[0]
                  or intersection_over_union(
                    torch.tensor(chosen_box[2:]),
                    torch.tensor(box[2:]),
                    box_format=box_format)
                  < iou_threshold]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

if __name__ == '__main__':
    import cProfile
    import pstats

    test_boxes = [
        [1, 1, 0.5, 0.45, 0.4, 0.5],
        [2, 0.9, 0.5, 0.5, 0.2, 0.4],
        [1, 0.8, 0.25, 0.35, 0.3, 0.1],
        [1, 0.05, 0.1, 0.1, 0.1, 0.1],
    ]

    c_boxes = [
        [1, 1, 0.5, 0.45, 0.4, 0.5],
        [2, 0.9, 0.5, 0.5, 0.2, 0.4],
        [1, 0.8, 0.25, 0.35, 0.3, 0.1],
    ]
    bboxes = non_max_suppression(test_boxes, iou_threshold=7/20, prob_threshold=0.2, box_format="midpoint")
    assert (sorted(bboxes) == sorted(c_boxes)), "do not correct"

    with cProfile.Profile() as pr:
        non_max_suppression(test_boxes, iou_threshold=7 / 20, prob_threshold=0.2, box_format="midpoint")

    stats = pstats.Stats(pr)

    stats.sort_stats(pstats.SortKey.TIME)
    # stats.print_stats()
    stats.dump_stats(filename="nms_profiling.prof")





