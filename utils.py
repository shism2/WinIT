import numpy as np


def imp_ft_within_ts_acc(identified_imp, gt_imp):
    """% of the time that for a given timestep, the most important feature has the highest saliency"""
    assert identified_imp.shape == gt_imp.shape
    assert len(identified_imp.shape) == 3, "(samples, fts, ts)"
    assert identified_imp.shape[1] > 1, "(>1 ft required)"

    important_ts = np.argwhere(np.max(gt_imp, axis=1))
    correct_count = 0
    for (i, t) in important_ts:
        # Ensure that there is only 1 imp ft
        ranked_gt_fts = np.sort(gt_imp[i, :, t])
        assert ranked_gt_fts[-1] > ranked_gt_fts[-2] == 0, "Should only be 1 imp ft per ts for now"

        # Ensure that the identified most important ft is unique
        ranked_fts = np.sort(identified_imp[i, :, t])
        if ranked_fts[-1] > ranked_fts[-2]:
            gt_imp_ft = np.argmax(gt_imp[i, :, t])
            identified_imp_ft = np.argmax(identified_imp[i, :, t])
            correct_count += gt_imp_ft == identified_imp_ft
    imp_ft_acc = correct_count / len(important_ts)
    return imp_ft_acc
