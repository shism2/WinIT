import numpy as np


def imp_ft_within_ts(identified_imp, gt_imp):
    """% of the time that for a given timestep, the most important feature has the highest saliency"""
    assert identified_imp.shape == gt_imp.shape
    assert len(identified_imp.shape) == 3, "(samples, fts, ts)"
    assert identified_imp.shape[1] > 1, "(>1 ft required)"

    identified_imp = np.nan_to_num(identified_imp)
    # TODO: Does this assumption about importance make sense
    identified_imp = np.abs(identified_imp)

    std_thresh = 1

    ranked_acc_count = 0
    above_margin_count = 0

    important_ts = np.argwhere(np.max(gt_imp, axis=1))
    for (i, t) in important_ts:
        # Ensure that there is only 1 imp ft
        ranked_gt_fts = np.sort(gt_imp[i, :, t])
        assert ranked_gt_fts[-1] > ranked_gt_fts[-2] == 0, "Should only be 1 imp ft per ts for now"

        # Ensure that the identified most important ft is unique
        ranked_fts = np.sort(identified_imp[i, :, t])
        if ranked_fts[-1] > ranked_fts[-2]:
            gt_imp_ft = np.argmax(gt_imp[i, :, t])
            identified_imp_ft = np.argmax(identified_imp[i, :, t])

            if gt_imp_ft == identified_imp_ft:
                # Ranked accuracy
                ranked_acc_count += 1

                # Imp ft is at least n std larger than unimp ft
                above_margin_count += ranked_fts[-1] - ranked_fts[-2] >= np.std(identified_imp[i]) * std_thresh

    print('-----------------------------------------------')
    print('Important features within each timestep metrics')

    imp_ft_acc = ranked_acc_count / len(important_ts)
    print('Ranked accuracy:', imp_ft_acc)

    avg_above_margin = above_margin_count / len(important_ts)
    print(f'Imp ft at least {std_thresh} std > than unimp ft: {avg_above_margin}')
    print('-----------------------------------------------')
    return imp_ft_acc, avg_above_margin
