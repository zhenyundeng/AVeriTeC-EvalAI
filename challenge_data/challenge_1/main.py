import random
# from averitec_scorer import averitec_score
import jsonlines
import json
import numpy as np
import scipy
import sklearn
import nltk
from nltk import word_tokenize


def evaluate_ori(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            "status": u"running",
            "when_made_public": None,
            "participant_team": 5,
            "input_file": "https://abc.xyz/path/to/submission/file.json",
            "execution_time": u"123",
            "publication_url": u"ABC",
            "challenge_phase": 1,
            "created_by": u"ABC",
            "stdout_file": "https://abc.xyz/path/to/stdout/file.json",
            "method_name": u"Test",
            "stderr_file": "https://abc.xyz/path/to/stderr/file.json",
            "participant_team_name": u"Test Team",
            "project_url": u"http://foo.bar",
            "method_description": u"ABC",
            "is_public": False,
            "submission_result_file": "https://abc.xyz/path/result/file.json",
            "id": 123,
            "submitted_at": u"2017-03-20T19:22:03.880652Z",
        }
    """
    output = {}
    print(kwargs["submission_metadata"])
    annotations = []
    with jsonlines.open(test_annotation_file) as f:
        for i, line in enumerate(f.iter()):
            annotation = {}
            annotation['label'] = line['label']
            annotation['evidence'] = line['evidence']
            annotations.append(annotation)
    with jsonlines.open(user_submission_file) as f:
        for i, line in enumerate(f.iter()):
            annotations[i]['predicted_label'] = line['predicted_label']
            annotations[i]['predicted_evidence'] = line['predicted_evidence']
    # strict_score, label_accuracy, precision, recall, f1 = averitec_score(annotations)
    label_accuracy, evidence_meteor = averitec_score(annotations)

    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "dev_split": {
                    "Label_accuracy": label_accuracy,
                    "Evidence_meteor": evidence_meteor,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["dev_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "Label_accuracy": label_accuracy,
                    "Evidence_meteor": evidence_meteor,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test"]
        print("Completed evaluation for Test Phase")
    # elif phase_codename == "test":
    #     print("Evaluating for Test Phase")
    #     output["result"] = [
    #         {
    #             "train_split": {
    #                 "Metric1": random.randint(0, 99),
    #                 "Metric2": random.randint(0, 99),
    #                 "Metric3": random.randint(0, 99),
    #                 "Total": random.randint(0, 99),
    #             }
    #         },
    #         {
    #             "test_split": {
    #                 "Metric1": random.randint(0, 99),
    #                 "Metric2": random.randint(0, 99),
    #                 "Metric3": random.randint(0, 99),
    #                 "Total": random.randint(0, 99),
    #             }
    #         },
    #     ]
    #     # To display the results in the result file
    #     output["submission_result"] = output["result"][0]
    #     print("Completed evaluation for Test Phase")
    return output


def compute_all_pairwise_scores(src_data, tgt_data, metric):
    X = np.empty((len(src_data), len(tgt_data)))

    for i in range(len(src_data)):
        for j in range(len(tgt_data)):
            X[i][j] = (metric(src_data[i], tgt_data[j]))

    return X


def averitec_score(predictions, actual=None, max_evidence=5, max_evidence_cell=25):
    #
    score_label = 0
    score_pairwise_evidence = 0

    for idx, instance in enumerate(predictions):
        # label
        gold_label = instance['label']
        pred_label = instance['predicted_label']

        if gold_label == pred_label:
            score_label += 1

        # evidence
        gold_evi = instance['evidence']
        pred_evi = instance['predicted_evidence']

        def pairwise_meteor(candidate, reference):  # Todo this is not thread safe, no idea how to make it so
            return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))

        pairwise_scores = compute_all_pairwise_scores(pred_evi, gold_evi, pairwise_meteor)
        assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
        assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()
        reweight_term = 1 / float(len(pred_evi))
        assignment_utility *= reweight_term
        score_pairwise_evidence += assignment_utility

    label_accuracy, evidence_meteor = score_label / len(predictions), score_pairwise_evidence / len(predictions)
    return label_accuracy, evidence_meteor


def evaluate(test_annotation_file, user_submission_file, phase_codename, **kwargs):
    print("Starting Evaluation.....")
    print("Submission related metadata:")
    """
    Evaluates the submission for a particular challenge phase adn returns score
    Arguments:

        `test_annotations_file`: Path to test_annotation_file on the server
        `user_submission_file`: Path to file submitted by the user
        `phase_codename`: Phase to which submission is made

        `**kwargs`: keyword arguments that contains additional submission
        metadata that challenge hosts can use to send slack notification.
        You can access the submission metadata
        with kwargs['submission_metadata']

        Example: A sample submission metadata can be accessed like this:
        >>> print(kwargs['submission_metadata'])
        {
            "status": u"running",
            "when_made_public": None,
            "participant_team": 5,
            "input_file": "https://abc.xyz/path/to/submission/file.json",
            "execution_time": u"123",
            "publication_url": u"ABC",
            "challenge_phase": 1,
            "created_by": u"ABC",
            "stdout_file": "https://abc.xyz/path/to/stdout/file.json",
            "method_name": u"Test",
            "stderr_file": "https://abc.xyz/path/to/stderr/file.json",
            "participant_team_name": u"Test Team",
            "project_url": u"http://foo.bar",
            "method_description": u"ABC",
            "is_public": False,
            "submission_result_file": "https://abc.xyz/path/result/file.json",
            "id": 123,
            "submitted_at": u"2017-03-20T19:22:03.880652Z",
        }
    """
    output = {}
    # print(kwargs["submission_metadata"])
    # annotations = []
    #
    # with jsonlines.open(test_annotation_file) as f:
    #     for i, line in enumerate(f.iter()):
    #         annotation = {}
    #         annotation['label'] = line['label']
    #         annotation['evidence'] = line['evidence']
    #         annotations.append(annotation)
    # with jsonlines.open(user_submission_file) as f:
    #     for i, line in enumerate(f.iter()):
    #         annotations[i]['predicted_label'] = line['predicted_label']
    #         annotations[i]['predicted_evidence'] = line['predicted_evidence']

    # test_samples = json.load(open(test_annotation_file, 'r'))
    # pred_samples = json.load(open(user_submission_file, 'r'))

    test_samples = [json.loads(l) for l in open(test_annotation_file, 'r').readlines()]
    pred_samples = [json.loads(l) for l in open(user_submission_file, 'r').readlines()]

    assert len(test_samples) == len(pred_samples)
    #
    # annotations = []
    # for idx, sample in enumerate(test_samples):
    #     annotation = {}
    #     annotation['label'] = sample['label']
    #     annotation['evidence'] = [sample['evidence']]
    #     # annotation['evidence'] = sample['evidence']
    #
    #     pred_sample = pred_samples[idx]
    #     annotation['predicted_label'] = pred_sample['label']
    #     annotation['predicted_evidence'] = [pred_sample['evidence']]
    #     # annotation['predicted_evidence'] = sample['evidence']
    #     annotations.append(annotation)
    #
    # with open("../../annotations/averitec_dev_gold.jsonl", 'w') as f:
    #     for x in annotations:
    #         f.write(json.dumps(x) + '\n')

    annotations = []
    for idx, sample in enumerate(test_samples):
        annotation = {}
        annotation['label'] = sample['label']
        annotation['evidence'] = sample['evidence']

        pred_sample = pred_samples[idx]
        annotation['predicted_label'] = pred_sample['label']
        annotation['predicted_evidence'] = pred_sample['evidence']

        annotations.append(annotation)

    label_accuracy, evidence_meteor = averitec_score(annotations)

    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "dev_split": {
                    "Label_accuracy": label_accuracy,
                    "Evidence_meteor": evidence_meteor,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["dev_split"]
        print("Completed evaluation for Dev Phase")
    elif phase_codename == "test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "Label_accuracy": label_accuracy,
                    "Evidence_meteor": evidence_meteor,
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test"]
        print("Completed evaluation for Test Phase")

    return output


# def compute_all_pairwise_scores(src_data, tgt_data, metric):
#     X = np.empty((len(src_data), len(tgt_data)))
#
#     for i in range(len(src_data)):
#         for j in range(len(tgt_data)):
#             X[i][j] = (metric(src_data[i], tgt_data[j]))
#     return X
#
#

if __name__ == "__main__":
    test_annotation_file = "../../dev.json"
    user_submission_file = "../../dev.json"

    evaluate(test_annotation_file, user_submission_file, 'dev')
    print("hello")

#
#     test_samples = json.load(open(test_annotation_file, 'r'))  # gold
#     pred_samples = json.load(open(user_submission_file, 'r'))
#
#     #
#     score_label = 0
#     score_pairwise_evidence = 0
#     #
#     annotations = []
#     for idx, sample in enumerate(test_samples):
#         annotation = {}
#         annotation['label'] = sample['label']
#         annotation['justification'] = [sample['justification']]
#         # annotation['evidence'] = sample['evidence']
#
#         pred_sample = pred_samples[idx]
#         annotation['predicted_label'] = pred_sample['label']
#         annotation['predicted_justification'] = [pred_sample['justification']]
#         # annotation['predicted_evidence'] = sample['evidence']
#         annotations.append(annotation)
#
#         # label
#         if sample['label'] == pred_sample['label']:
#             score_label += 1
#
#
#         # evidence
#         def pairwise_meteor(candidate, reference):  # Todo this is not thread safe, no idea how to make it so
#             return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))
#
#
#         pairwise_scores = compute_all_pairwise_scores(annotation['predicted_justification'],
#                                                       annotation['justification'], pairwise_meteor)
#         assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
#         assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()
#         reweight_term = 1 / float(len(annotation['predicted_justification']))
#         assignment_utility *= reweight_term
#         score_pairwise_evidence += assignment_utility
#
#     # label
#     acc = score_label / len(test_samples)
#     # evidence
#     evi_score = score_pairwise_evidence / len(test_samples)
