import numpy as np
import re
from nltk.corpus import stopwords
from sklearn.svm import SVC
from classifier import run_classifier

dimensions = {
    "HarmVirtue": ["safe", "peace", "compassion", "empath", "sympath", "care", "caring", "protect", "shield", "shelter", "amity", "secur", "benefit", "defen", "guard", " preserve"],
    "HarmVice": ["harm", "suffer", "war", "wars", "warl", "warring", "fight", "violen", "hurt", "kill", "kills", "killer", "killed", "killing", "endanger", "cruel", "brutal", "abuse", "damage", "ruin", "ravage", "detriment", "crush", "attack", "annihilate", "destroy", "stomp", "abandon", "spurn", "impair", "exploit", "exploits", "exploited", "exploiting", "wound"],
    "FairnessVirtue": [
        "fair", "fairly", "fairness", "fair-", "fairmind", "fairplay", "equal", "justice", "justness", "justifi", "reciproc", "impartial", "egalitar", "rights", "equity", "eveness", "equivalent", "unbias", "tolerant", "equable", "balance", "unprejudice", "reasonable", "constant", "honest"],
    "FairnessVice": [
        "unfair", "unequal", "bias", "unjust", "injust", "bigot", "discriminate", "disproportion", "inequitable", "prejud", "dishonest", "unscrupulous", "dissociate", "preference", "favoritism", "segregat", "exclusion", "exclud"],
    "IngroupVirtue": ["segregat", "together", "nation", "homeland", "family", "families", "familial", "group", "loyal", "patriot", "communal", "commune", "communit", "communis", "comra", "cadre", "collectiv", "join", "unison", "unite", "fellow", "guild", "solidarity", "devot", "member", "cliqu", "cohort", "ally", "insider"],
    "IngroupVice": ["abandon", "foreign",
                    "enem", "betray", "treason", "traitor", "treacher", "disloyal", "individual", "apostasy", "apostate", "deserter", "deserting", "deceiv", "jilt", "imposter", "miscreant", "spy", "sequester", "renegade", "terroris", "immigra"],
    "AuthorityVirtue": ["preserve", "loyal",
                        "obey", "obedien", "duty", "law", "lawful", "legal", "duti", "honor", "respect", "respectful", "respects", "respected", "order", "father", "mother", "motherl", "mothering", "mothers",  "tradition", "hierarch", "authorit", "permit", "permission", "status", "rank", "leader", "class", "bourgeoisie", "caste", "position", "complian", "command", "supremacy", "control", "submit", "allegian", "serve", "abide", "defere", "defer", "revere", "venerat", "comply"],
    "AuthorityVice": ["betray", "treason", "traitor", "treacher", "disloyal", "apostasy", "apostate", "deserter", "deserting",
                      "defian", "rebel", "dissent", "subver", "disrespect", "disobe", "sediti", "agitat", "insubordinat", "illegal", "lawless", "insurgent", "mutinous", "defy", "dissident", "unfaithful", "alienate", "defector", "heretic", "nonconformist", "oppose", "protest", "refuse", "denounce", "remonstrate", "riot", "obstruct"],
    "PurityVirtue": ["preserve",
                     "piety", "pious", "purity", "pure", "clean", "steril", "sacred", "chast", "holy", "holiness", "saint", "wholesome", "celiba", "abstention", "virgin", "virgins", "virginity", "virginal", "austerity", "integrity", "modesty", "abstinen", "abstemiousness", "upright", "limpid", "unadulterated", "maiden", "virtuous", "refined", "decent", "immaculate", "innocent", "pristine", "church"],
    "PurityVice": ["ruin", "exploit", "exploits", "exploited", "exploiting", "apostasy", "apostate", "heretic",

                   "disgust", "deprav", "diseas", "unclean", "contagio", "indecen", "sin", "sinful", "sinner", "sins", "sinned", "sinning", "slut", "whore", "dirt", "impiety", "impious", "profan", "gross", "repuls", "sick", "promiscu", "lewd", "adulter", "debauche", "defile", "tramp", "prostitut", "unchaste", "intemperate", "wanton", "profligate", "filth", "trashy", "obscen", "lax", "taint", "stain", "tarnish", "debase", "desecrat", "wicked", "blemish", "explotat", "pervert", "wretched"],
    "MoralityGeneral": ["honest", "lawful", "legal", "piety", "pious", "wholesome", "integrity",  "upright", "decen", "indecen", "wicked", "wretched",
                        "righteous", "moral", "ethic", "value", "upstanding", "good", "goodness", "principle", "blameless", "exemplary", "lesson", "canon", "doctrine", "noble", "worth", "ideal", "praiseworthy", "commendable", "character", "proper", "laudable", "correct", "wrong", "evil", "immoral", "bad", "offend", "offensive", "transgress"]
}


def moral_foundations(X_train, X_test, y_train, y_true):
    def feature(data):
        data = data['request_text']
        return calc_features(data)

    train_narr_feature = feature(X_train)
    test_narr_feature = feature(X_test)
    c = SVC(kernel='linear', probability=True)

    run_classifier(c, train_narr_feature, test_narr_feature, y_train, y_true)


def calc_features(data):
    dimension_features = []

    for row in data:
        row_vector = []
        white_spaced_words = len(row.split())

        if white_spaced_words:
            for dim in dimensions:
                dim_words = dimensions[dim]
                score = 0

                for word in dim_words:
                    score += 1 if re.search(r'\b%s\w*' %
                                            word, row, flags=re.IGNORECASE) else 0

                row_vector.append(score/white_spaced_words)
        else:
            row_vector = [0] * 11

        dimension_features.append(row_vector)

    return dimension_features
