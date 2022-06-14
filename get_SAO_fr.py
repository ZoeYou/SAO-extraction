from turtle import clear
import spacy 
import re

# !python -m spacy download fr_dep_news_trf
nlp = spacy.load('fr_dep_news_trf')
stopwords = nlp.Defaults.stop_words
stopwords.add("moyens")


def find_mod_subj(token, doc):
    """
    given the subject token, returns its noun trunk
    """
    idx = [token.i]

    amod_tokens = [child for child in token.children if child.dep_=="amod"]
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
    return doc[min(idx): max(idx)+1]

def find_mod_obj(token):
    """
    given the object token, returns it noun trunk
    """
    idx = [token.i]
    amod_tokens = [child for child in token.children if child.dep_=="amod"]
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
    subject_trunk = subject_trunk.text
    object_trunk = object_trunk.text

    if (subject_trunk in stopwords or object_trunk in stopwords) or (subject_trunk==object_trunk):
        ret = False
    if subject_trunk.isdigit() or object_trunk.isdigit():
        ret = False
    return ret

def clean_trunk(noun_trunk):
    noun_trunk = re.sub("^[1-9]*\.", "", noun_trunk)

    tokens = noun_trunk.split(" ")
    if tokens[0] == "dont":
        tokens = tokens[1:]
    return " ".join(tokens)

def get_SAO_fr(sentence, model=nlp):
    doc = model(sentence)
    res = []

    # filter for obj
    nouns_obj = [n for n in doc if n.dep_=="obj"]

    for n in nouns_obj:
        if n.head.pos_ == "VERB":
            verb = n.head
            object = n 
        else:
            continue

        subject = None
        # situation 1: acl 
        if n.head.dep_ == "acl":
            subject = n.head.head
            while subject.dep_ == "acl":
                subject = subject.head

        # situation 2: nsubj
        else:
            subject_node = [child for child in n.head.children if child.dep_=="nsubj"]
            if subject_node: subject = subject_node[0]
        
        # find noun trunk that includes the subject noun
        if subject:
            subject_trunk = find_mod_subj(subject, doc)
            
            obj_start, obj_end = find_mod_obj(object)
            object_trunk = doc[obj_start: obj_end+1]
            if is_valid(subject_trunk, object_trunk):

                res.append((clean_trunk(subject_trunk.text), verb.lemma_, clean_trunk(object_trunk.text)))

            # check if object has another conjunct object
            conj_tokens = [child for child in object_trunk.root.children if child.dep_=="conj"]
            if conj_tokens:
                for t in conj_tokens:
                    obj_start, obj_end = find_mod_obj(t)
                    object_trunk = doc[obj_start: obj_end+1]
                    if is_valid(subject_trunk, object_trunk):
                        res.append((clean_trunk(subject_trunk.text), verb.lemma_, clean_trunk(object_trunk.text)))
        else:
            continue
    return res

############################ TEST ################################
from nltk import sent_tokenize

claims = "REVENDICATIONS1. Conduite de transport d'un liquide lave-glace pour véhicule automobile comportant au moins un canal de circulation du liquide lave glace destiné à s'étendre depuis un réservoir de liquide lave glace jusqu'à un balai d'essuie-glace, caractérisée en ce qu'elle comporte d'au moins un élément filaire conducteur électrique (20,25,50,75) comportant au moins une âme électrique centrale (21,26,51,76) et un brin électrique coaxial (23,28,53,78) qui est isolé électriquement de la dit âme centrale (21,26,51,76). 2. Conduite selon la revendication 1, caractérisée en ce que le brin électrique coaxial (23,28,53,78) est bobiné en hélice autour d'un manchon électriquement isolant (22,27,52,77) entourant l'âme électrique centrale (21,26,51,76). 3. Conduite selon l'une quelconque des revendications 1 à 2, caractérisée en ce que l'âme centrale électrique (21,26,51,76) est tressée. 4. Conduite selon l'une quelconque des revendications 1 et 2, caractérisée en ce que l'élément filaire conducteur électrique (20,25,50,75) comporte une gaine extérieure isolante (24,29,54,79). 5. Conduite selon l'une quelconque des revendications 1 à 4, caractérisée en ce qu'elle comporte un premier élément filaire conducteur électrique (20) qui assure l'alimentation électrique d'un dispositif de chauffe (40,68) d'un balai d'essuie-glace, dont une première extrémité (37) est alimentée en électricité et dont l'extrémité opposée (39) est reliée électriquement au balai d'essuie-glace. 6. Conduite selon l'une quelconque des revendications 1 à 4, caractérisée en ce qu'elle comporte un deuxième élément filaire conducteur électrique (25) qui est chauffant, qui assure le chauffage du liquide lave glace circulant dans le canal de circulation (32 ;35,36), dont une première extrémité (42) est alimentée en électricité et dont l'extrémité opposée (43) comporte des moyens (44) pour relier 3039483 12 électriquement l'âme centrale (26) au brin coaxial (28) en formant une boucle de courant. 7. Conduite selon l'une quelconque des revendications 1 à 4, caractérisée en 5 ce qu'elle comporte un premier élément filaire conducteur électrique (20) qui assure l'alimentation électrique d'un dispositif de chauffe du balai d'essuie-glace, dont une première extrémité (37) est alimentée en électricité et dont l'extrémité opposée (39) alimente le dispositif de chauffe du balai d'essuie-glace, et en ce qu'elle comporte en outre un deuxième élément filaire conducteur électrique (25) 10 qui est chauffant, qui assure le chauffage du liquide lave glace circulant dans le canal de circulation, dont une première extrémité (42) est alimentée en électricité et dont l'extrémité opposée (43) comporte des moyens pour relier l'âme centrale (26) au brin coaxial (28) en formant une boucle de courant. 15 8. Conduite selon l'une quelconque des revendications 1 à 4, caractérisée en ce qu'elle comporte une paire de troisièmes éléments filaires conducteurs électriques (50,50a,50b) dont le brin coaxial électrique (53,53a) est chauffant, en ce que les brins coaxiaux électriques chauffants (53a,53b) desdits troisièmes éléments filaires (50a,50b) sont d'une part alimentés électriquement et d'autre 20 part connectés entre eux pour former une boucle de courant et assurer le chauffage du liquide lave glace circulant dans le canal de circulation (57 ;62,63), et en ce que l'âme électrique centrale (51a) de chacun des troisièmes éléments filaires (50a,50b) est d'une part alimentée électriquement et d'autre part électriquement connectée à un dispositif de chauffe du balai d'essuie-glace. 25 9. Conduite selon l'une quelconque des revendications 1 à 4, caractérisée en ce qu'elle comporte une paire de quatrièmes éléments filaires conducteurs électriques (75) dont l'âme électrique centrale (78) est chauffante, en ce que les âmes électriques centrales chauffantes (78) des quatrièmes éléments filaires (75) 30 sont respectivement d'une part alimentées électriquement et d'autre part connectées entre elles pour former une boucle de courant et assurer le chauffage du liquide lave glace circulant dans le canal de circulation (57 ;62,63), et en ce que le brin coaxial électrique (76) de chacun des quatrièmes éléments filaires 3039483 13 (75) est d'une part alimenté électriquement et d'autre part électriquement connecté à un dispositif de chauffe du balai d'essuie-glace. 5 10. Conduite selon l'une quelconque des revendications précédentes, caractérisée en ce qu'elle comporte un manchon extrudé (31,34,56,61,46) comportant au moins un canal de circulation du liquide lave glace (32 ;35,36 ;57 ;62,63 ;47,48), et en ce qu'au moins un élément filaire conducteur électrique (20 ;25 ;50,75) est noyé dans la masse du dit manchon 10 (31,34,56,61,46) . 11. Conduite selon l'une quelconque des revendications précédentes, caractérisée en ce qu'elle comporte un canal de circulation de liquide lave-glace (32,57) et en ce que des éléments filaires conducteurs électriques 15 (20,50,50a,50b,75) sont disposés de part et d'autre du dit canal de circulation (32,57). 12. Conduite selon l'une quelconque des revendications 1 à 10, caractérisée en ce qu'elle comporte deux canaux de circulation (35,36 ;62,63 ;47,48) pour 20 alimenter un balai d'essuie-glace à deux rampes d'arrosage. 13. Conduite selon la revendication 12, caractérisé en ce que les élément filaires conducteurs électriques (25,50,50a,50b,75) sont disposés entre les deux canaux de circulation (35,36 ;62,63). 25 14.Dispositif d'essuyage pour surface vitrée de véhicule automobile comportant au moins un balai d'essuie-glace muni d'au moins une première rampe d'arrosage, caractérisé en ce qu'il comporte une conduite de transport d'un liquide lave glace (30,33,55,60,45) selon l'une quelconque des revendications 1 à 30 13, raccordée à ladite première rampe d'arrosage. 15. Dispositif d'essuyage pour surface vitrée de véhicule automobile comportant au moins un balai d'essuie-glace muni d'une première et d'une seconde rampe d'arrosage de chaque côté du bras du balai d'essuie-glace, 3039483 -14 caractérisé en ce qu'il comporte une conduite de transport d'un liquide lave glace (33,60,45) selon l'une quelconque des revendications 12 et 13."

sents = sent_tokenize(claims, language="french")
for sent in sents:
    saos = get_SAO_fr(sent)
    if saos:
        print(saos)