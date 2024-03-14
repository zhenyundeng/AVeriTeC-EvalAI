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
    score_label1, score_label2, score_label3 = 0, 0, 0
    score_evidence, score_evidence1, score_evidence2, score_evidence3 = 0, 0, 0, 0

    for idx, instance in enumerate(predictions):
        # if idx == 321:
        #     print('hello')
        # label
        gold_label = instance['label']
        pred_label = instance['predicted_label']

        # evidence
        gold_evi = instance['evidence']
        pred_evi = instance['predicted_evidence']

        def pairwise_meteor(candidate, reference):  # Todo this is not thread safe, no idea how to make it so
            return nltk.translate.meteor_score.single_meteor_score(word_tokenize(reference), word_tokenize(candidate))

        if pred_label and pred_evi:
            pairwise_scores = compute_all_pairwise_scores(pred_evi, gold_evi, pairwise_meteor)
            assignment = scipy.optimize.linear_sum_assignment(pairwise_scores, maximize=True)
            assignment_utility = pairwise_scores[assignment[0], assignment[1]].sum()
            reweight_term = 1 / float(len(pred_evi))
            assignment_utility *= reweight_term
            # score_evidence += assignment_utility

            if assignment_utility >= 0.20:
                if gold_label == pred_label:
                    score_label1 += 1
                score_evidence1 += assignment_utility
            if assignment_utility >= 0.25:
                if gold_label == pred_label:
                    score_label2 += 1
                score_evidence2 += assignment_utility
            if assignment_utility >= 0.30:
                if gold_label == pred_label:
                    score_label3 += 1
                score_evidence3 += assignment_utility

    label_accuracy1, evidence_meteor1 = score_label1 / len(predictions), score_evidence1 / len(predictions)
    label_accuracy2, evidence_meteor2 = score_label2 / len(predictions), score_evidence2 / len(predictions)
    label_accuracy3, evidence_meteor3 = score_label3 / len(predictions), score_evidence3 / len(predictions)

    return [label_accuracy1, label_accuracy2, label_accuracy3], [evidence_meteor1, evidence_meteor2, evidence_meteor3]
    # return label_accuracy1, evidence_meteor1, label_accuracy2, evidence_meteor2, label_accuracy3, evidence_meteor3


def extract_evidence_from_sample(sample):
    example_strings = []
    url_strings = []

    if "questions" in sample:
        for idx, question in enumerate(sample['questions']):
            if not isinstance(question["answers"], list):
                question["answers"] = [question["answers"]]

            for answer in question["answers"]:
                example_strings.append(question["question"] + " " + answer["answer"])
                if "answer_type" in answer and answer["answer_type"] == "Boolean":
                    example_strings[-1] += ". " + answer["boolean_explanation"]

                url_strings.append(answer["source_url"])

            if len(question["answers"]) == 0:
                example_strings.append(question["question"] + " No answer could be found.")
                url_strings.append("No URL could be found.")

    if "string_evidence" in sample:
        for full_string_evidence in sample["string_evidence"]:
            example_strings.append(full_string_evidence)

    assert len(url_strings) == len(example_strings)

    return example_strings


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

    test_samples = [json.loads(l) for l in open(test_annotation_file, 'r').readlines()]
    pred_samples = [json.loads(l) for l in open(user_submission_file, 'r').readlines()]
    assert len(test_samples) == len(pred_samples)

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
                    "Label accuracy": label_accuracy[1],
                    "Evidence meteor": evidence_meteor[1],
                    # "Label accuracy1": label_accuracy[0],
                    # "Evidence meteor1": evidence_meteor[0],
                    # "Label accuracy2": label_accuracy[1],
                    # "Evidence meteor2": evidence_meteor[1],
                    # "Label accuracy3": label_accuracy[2],
                    # "Evidence meteor3": evidence_meteor[2],
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
                    "Label accuracy": label_accuracy[1],
                    "Evidence meteor": evidence_meteor[1],
                    # "Label accuracy1": label_accuracy[0],
                    # "Evidence meteor1": evidence_meteor[0],
                    # "Label accuracy2": label_accuracy[1],
                    # "Evidence meteor2": evidence_meteor[1],
                    # "Label accuracy3": label_accuracy[2],
                    # "Evidence meteor3": evidence_meteor[2],
                }
            }
        ]
        # To display the results in the result file
        output["submission_result"] = output["result"][0]["test"]
        print("Completed evaluation for Test Phase")

    return output


def create_gold_file():

    # dev_annotation_file = "dev.json"   # "../../dev.json"
    test_annotation_file = "baseline_test.json"  # "test.json", baseline_test.json

    # dev_samples = json.load(open(dev_annotation_file, 'r'))
    test_samples = json.load(open(test_annotation_file, 'r'))

    # # ----------
    # # averitec_dev_gold.jsonl
    # annotations_dev = []
    # for idx, sample in enumerate(dev_samples):
    #     annotation = {}
    #     annotation['id'] = idx
    #     annotation['label'] = sample['label']
    #
    #     evidences = extract_evidence_from_sample(sample)
    #     annotation['evidence'] = evidences
    #     annotations_dev.append(annotation)
    #
    # with open("annotations/averitec_dev_gold.jsonl", 'w') as f:
    #     for x in annotations_dev:
    #         f.write(json.dumps(x) + '\n')
    # # ----------

    # ----------
    # averitec_test_gold.jsonl
    annotations_test = []
    for idx, sample in enumerate(test_samples):
        annotation = {}
        annotation['id'] = idx
        annotation['label'] = sample['label']

        evidences = extract_evidence_from_sample(sample)
        annotation['evidence'] = evidences
        annotations_test.append(annotation)

    with open("annotations/averitec_test_gold.jsonl", 'w') as f:
        for x in annotations_test:
            f.write(json.dumps(x) + '\n')
    # ----------

    return 0


# if __name__ == "__main__":
#
#     # create_gold_file()  # averitec_dev_gold.jsonl, averitec_test_gold.jsonl
#
#     test_annotation_file = "annotations/averitec_test_gold.jsonl"
#     user_submission_file = "baseline_submission_test.jsonl"
#
#     evaluate(test_annotation_file, user_submission_file, 'dev')
#     print("hello")

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
