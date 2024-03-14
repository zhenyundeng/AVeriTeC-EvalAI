import json
import numpy as np
import scipy
import nltk
from nltk import word_tokenize
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

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
        # label
        gold_label = instance['label']
        pred_label = instance['predicted_label']
        # evidence
        gold_evi = instance['evidence']
        pred_evi = instance['predicted_evidence']

        if gold_label == pred_label:
            score_label1 += 1

    label_accuracy = score_label1 / len(predictions)
    return label_accuracy


def averitec_score1(predictions, actual=None, max_evidence=5, max_evidence_cell=25):
    #
    score_label1, score_label2, score_label3 = 0, 0, 0
    score_evidence, score_evidence1, score_evidence2, score_evidence3 = 0, 0, 0, 0

    for idx, instance in enumerate(predictions):
        # label
        gold_label = instance['label']
        pred_label = instance['predicted_label']
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
    #
    annotations = []
    for idx, sample in enumerate(test_samples):
        annotation = {}
        annotation['label'] = sample['label']
        annotation['evidence'] = sample['evidence']

        pred_sample = pred_samples[idx]
        annotation['predicted_label'] = pred_sample['label']
        annotation['predicted_evidence'] = pred_sample['evidence']

        annotations.append(annotation)

    label_accuracy = averitec_score(annotations)
    # label_accuracy, evidence_meteor = averitec_score(annotations)

    if phase_codename == "dev":
        print("Evaluating for Dev Phase")
        output["result"] = [
            {
                "dev_split": {
                    "Label accuracy": label_accuracy,
                    # "Evidence meteor": evidence_meteor[1],
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
    elif phase_codename == "test" or phase_codename == "after_test":
        print("Evaluating for Test Phase")
        output["result"] = [
            {
                "test_split": {
                    "Label accuracy": label_accuracy,
                    # "Evidence meteor": evidence_meteor[1],
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


# if __name__ == "__main__":
#     test_annotation_file = "annotations/averitec_dev_gold.jsonl"
#     user_submission_file = "baseline_submission_dev.jsonl"
#
#     evaluate(test_annotation_file, user_submission_file, 'dev')
#     print("hello")
