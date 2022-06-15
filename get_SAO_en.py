import spacy 

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

with open("US_stopwords.txt") as inf:
    stopwords_to_append = inf.read().splitlines()
stopwords.update(stopwords_to_append)

def get_SAO_en(sentence, model=nlp):
    doc = model(sentence)
    res = []

    # get all noun trunks 
    nouns = [n for n in doc.noun_chunks]

    # filter for dobj
    nouns_dobj = [n for n in nouns if n.root.dep_=="dobj"]

    for n in nouns_dobj:
        object = " ".join([w.text for w in n if w.pos_!="DET"])
        verb = n.root.head.lemma_

        subject = None
        # situation 1: acl 
        if n.root.head.dep_ == "acl":
            subject = n.root.head.head
            while subject.dep_ == "acl":
                subject = subject.head
        
        # situation 2: nsubj
        else:
            subject_node = [child for child in n.root.head.children if child.dep_=="nsubj"]
            if subject_node: subject = subject_node[0]
        
        # find noun trunk that includes the subject noun
        if subject:
            try:
                subject = [n for n in nouns if subject in n][0]
                subject = " ".join([w.text for w in subject if w.pos_!="DET"])
            except IndexError:
                subject = subject.text

            if (subject not in stopwords and object not in stopwords) and (subject!=object):
                res.append((subject, verb, object))
    return res