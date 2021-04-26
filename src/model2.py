import numpy as np
from nltk import FreqDist
from sklearn.linear_model import SGDClassifier
from classifier import run_classifier

flair_encoding = {
    "none": [0, 0, 0],
    "shroom": [0, 1, 0],
    "PIF": [1, 1, 1]
}

def activity_reputation(X_train, X_test, y_train, y_true):
    def feature(data):
        top_20_sr = top_subreddits(data)
        return calc_features(data, top_20_sr)

    train_act_rep_feature = feature(X_train)
    test_act_rep_feature = feature(X_test)
    c = SGDClassifier(loss='hinge')
    run_classifier(c, train_act_rep_feature, test_act_rep_feature, y_train, y_true)


def calc_features(data, top_20_sr):
    feature_vector = []

    for (_, row) in data.iterrows():
        row_vector = []

        row_vector.append(1 if row.loc['post_was_edited'] else 0)
        row_vector.append(row.loc['requester_account_age_in_days_at_request'])
        row_vector.append(
            row.loc['requester_account_age_in_days_at_retrieval'])
        row_vector.append(
            row.loc['requester_days_since_first_post_on_raop_at_request'])
        row_vector.append(
            row.loc['requester_days_since_first_post_on_raop_at_retrieval'])
        row_vector.append(row.loc['requester_number_of_comments_at_request'])
        row_vector.append(row.loc['requester_number_of_comments_at_retrieval'])
        row_vector.append(
            row.loc['requester_number_of_comments_in_raop_at_request'])
        row_vector.append(
            row.loc['requester_number_of_comments_in_raop_at_retrieval'])
        row_vector.append(row.loc['requester_number_of_posts_at_request'])
        row_vector.append(row.loc['requester_number_of_posts_at_retrieval'])
        row_vector.append(
            row.loc['requester_number_of_posts_on_raop_at_request'])
        row_vector.append(
            row.loc['requester_number_of_posts_on_raop_at_retrieval'])
        row_vector.append(row.loc['requester_number_of_subreddits_at_request'])
        row_vector.append(
            row.loc['requester_account_age_in_days_at_retrieval'])

        # one hot encode sr values
        post_author_subreddits = row.loc['requester_subreddits_at_request']
        onehot_sr = [0] * 10

        for sr in post_author_subreddits:
            if sr in top_20_sr:
                at = top_20_sr.index(sr)
                onehot_sr[at] = 1

        row_vector.extend(onehot_sr)
        row_vector.append(
            row.loc['number_of_downvotes_of_request_at_retrieval'])
        row_vector.append(row.loc['number_of_upvotes_of_request_at_retrieval'])
        row_vector.append(
            row.loc['requester_upvotes_minus_downvotes_at_request'])
        row_vector.append(
            row.loc['requester_upvotes_minus_downvotes_at_retrieval'])
        row_vector.append(
            row.loc['requester_upvotes_plus_downvotes_at_request'])

        # one hot encode flair
        flair = row.loc['requester_user_flair']
        row_vector.extend(flair_encoding[flair]
                          if flair else flair_encoding["none"])

        # update overall feature vector
        feature_vector.append(row_vector)

    return feature_vector


def top_subreddits(data):
    subreddits = []
    data['requester_subreddits_at_request'].apply(
        lambda row: subreddits.append(row) if row else None)

    subreddits = sum(subreddits, [])
    sr_freq = FreqDist(subreddits).most_common(10)
    top_20_sr = list(map(lambda freq: freq[0], sr_freq))

    return top_20_sr
