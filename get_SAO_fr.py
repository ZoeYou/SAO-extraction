from turtle import clear
import spacy 
import re

# !python -m spacy download fr_dep_news_trf
nlp = spacy.load('fr_dep_news_trf')
stopwords = nlp.Defaults.stop_words

with open("./fr_stopwords.txt") as inf:
   stopwords_to_append = inf.read().splitlines()
stopwords.update(stopwords_to_append)

KEEP_DETERMINER = True # Whether to keep determiners

def find_mod_subj(token):
    """
    given the subject token, returns its noun trunk
    """
    idx = [token.i]

    if KEEP_DETERMINER:
        amod_tokens = [child for child in token.children if child.dep_ in ["amod","appos",'det']]
    else:
        amod_tokens = [child for child in token.children if child.dep_ in ["amod","appos"]]

    if amod_tokens:
        idx.extend([t.i for t in amod_tokens])

    low, high = find_mod_obj(token)
    idx.extend([low, high])

    if token.head.dep_=="nmod":
        nmod_token = token.head
        idx.append(nmod_token.i)
        while nmod_token.dep_=="nmod":
            nmod_token = nmod_token.head
            idx.append(nmod_token.i)

    nmod_tokens = [child for child in token.children if child.dep_=="nmod"]
    if nmod_tokens:
        for t in nmod_tokens:
            index_low, index_high = find_mod_obj(t)
        idx.extend([index_low, index_high])

    return min(idx), max(idx)

def find_mod_obj(token):
    """
    given the object token, returns it noun trunk
    """
    idx = [token.i]
    if KEEP_DETERMINER:
        amod_tokens = [child for child in token.children if child.dep_ in ["det", "amod"]]
    else:
        amod_tokens = [child for child in token.children if child.dep_ == "amod"]

    if amod_tokens:
        idx.extend([t.i for t in amod_tokens])

    nmod_tokens = [child for child in token.children if child.dep_=="nmod"]
    if nmod_tokens:
        for t in nmod_tokens:
            index_low, index_high = find_mod_obj(t)
        idx.extend([index_low, index_high])

    return min(idx), max(idx)

def is_valid(subject_trunk, object_trunk):
    ret = True
    subject_trunk_text = subject_trunk.text
    object_trunk_text = object_trunk.text

    if (subject_trunk.root.text in stopwords or object_trunk.root.text in stopwords) or (subject_trunk_text==object_trunk_text):
        ret = False
    if not(bool(re.search("[a-zA-Zéàèùâêîôûçëïü]",subject_trunk_text)) and bool(re.search("[a-zA-Zéàèùâêîôûçëïü]",object_trunk_text))):
        ret = False
    if any(x in object_trunk_text for x in ["quelconque des revendications", "quelconque des revendications", "selon la revendication"]) or any(x in subject_trunk_text for x in ["quelconque des revendications", "quelconque des revendications", "selon la revendication"]):
        ret = False
    return ret

def clean_trunk(noun_trunk):
    noun_trunk = re.sub("^[1-9]*\.| \(.*[^)]$", "", noun_trunk)

    tokens = noun_trunk.split(" ")
    if tokens[0] == "dont":
        tokens = tokens[1:]
    return " ".join(tokens)

def get_SAO_fr(sentence, model=nlp):
    sentence = " ".join([token for token in re.split(":| ", sentence) if token!=""])
    doc = model(sentence)
    res = []

    # filter for obj
    nouns_obj = [n for n in doc if n.dep_=="obj"]

    for n in nouns_obj:
        if n.head.pos_ == "VERB":
            verbs = [n.head]
            object = n 
            
            #if exists conjuncts of principle verb 
            if n.head.dep_ == "conj" and n.head.head.pos_ == "VERB":
                verbs.append(n.head.head)
            verbs.extend([token for token in n.head.children if token.pos_ == "VERB" and token.dep_ == "conj"])
        else:
            continue

        for verb in verbs:
            subject = None
            # situation 1: acl
            if n.head.dep_ == "acl":
                subject = n.head.head
                while subject.dep_ == "acl":
                    subject = subject.head

            # situation 2: nsubj
            else:
                subject_node = [child for child in n.head.children if child.dep_=="nsubj"]
                if subject_node: 
                    subject = subject_node[0]
                    if subject.text == "qui" and n.head.dep_=="acl:relcl":
                        subject = n.head.head
            
            # find noun trunk that includes the subject noun
            if subject:
                subj_start, subject_end = find_mod_subj(subject)
                subject_trunk = doc[subj_start: subject_end+1]
                
                obj_start, obj_end = find_mod_obj(object)
                object_trunk = doc[obj_start: obj_end+1]
                if is_valid(subject_trunk, object_trunk):
                    res.append((clean_trunk(subject_trunk.text), verb.lemma_, clean_trunk(object_trunk.text)))

                # check if subject has another conjunct subject
                conj_subj_tokens = [child for child in subject_trunk.root.children if child.dep_=="conj"]
  
                for t in conj_subj_tokens:
                    subj_start, subject_end = find_mod_subj(t)
                    subject_trunk = doc[subj_start: subject_end+1]
                    if is_valid(subject_trunk, object_trunk):
                        res.append((clean_trunk(subject_trunk.text), verb.lemma_, clean_trunk(object_trunk.text)))

                # check if object has another conjunct object
                conj_obj_tokens = [child for child in object_trunk.root.children if child.dep_=="conj"]

                for t in conj_obj_tokens:
                    obj_start, obj_end = find_mod_obj(t)
                    object_trunk = doc[obj_start: obj_end+1]
                    if is_valid(subject_trunk, object_trunk):
                        res.append((clean_trunk(subject_trunk.text), verb.lemma_, clean_trunk(object_trunk.text)))
        else:
            continue
    return res
