from turtle import clear
import mlconjug3
from nltk import sent_tokenize
from get_SAO_fr import *

######################################## example from mlconjug3 ###################################
# # To use mlconjug3 with the default parameters and a pre-trained conjugation model.
default_conjugator = mlconjug3.Conjugator(language='fr')

# # Verify that the model works
# test1 = default_conjugator.conjugate("manger").conjug_info['Indicatif']['Présent']['1p']
# test2 = default_conjugator.conjugate("partir").conjug_info['Indicatif']['Présent']['1p']
# test3 = default_conjugator.conjugate("facebooker").conjug_info['Indicatif']['Présent']['1p']
# test4 = default_conjugator.conjugate("astigratir").conjug_info['Indicatif']['Présent']['1p']
# test5 = default_conjugator.conjugate("mythoner").conjug_info['Indicatif']['Présent']['1p']
# print(test1)
# print(test2)
# print(test3)
# print(test4)
# print(test5)

# # You can now iterate over all conjugated forms of a verb by using the newly added Verb.iterate() method.
# default_conjugator = mlconjug3.Conjugator(language='en')
# test_verb = default_conjugator.conjugate("be")
# all_conjugated_forms = test_verb.iterate()
# print(all_conjugated_forms)
##################################################################################################

claims = "REVENDICATIONS1. Combineur (10) pour afficheur tête haute, comprenant : - une première face ((Pi) destinée à recevoir un rayonnement lumineux généré par un dispositif de projection (20) ; et - une seconde face (02) opposée à la première face (01) ; caractérisé en ce qu'une lame quart d'onde (18) est interposée entre la première face (mi) et la seconde face (02). 2. Combineur selon la revendication 1, comprenant un insert biseauté (16). 3. Combineur selon la revendication 2, dans lequel la lame quart d'onde (18) est déposée sur l'insert biseauté (16). 4. Combineur selon la revendication 2 ou 3, dans lequel l'insert biseauté (16) est reçu entre deux lames de verre (12, 14). 5. Combineur selon la revendication 4, dans lequel la première face ((Pi) est formée sur l'une (14) desdites lames de verre et dans lequel la seconde face (02) est formée sur l'autre (12) desdites lames de verre. 6. Combineur selon la revendication 4 ou 5, dans lequel la lame quart d'onde (18) est reçue entre l'insert biseauté (16) et une (12) desdites lames de 25 verre. 7. Combineur selon l'une des revendications 1 à 6, formant un pare-brise (10). 30 8. Afficheur tête haute comprenant un dispositif de projection (20) et un combineur (10) selon l'une des revendications 1 à 7. 9. Procédé de fabrication d'un combineur (10) comprenant une première lame de verre (14) formant une première face ((Pi) destinée à recevoir un 3043798 8 rayonnement lumineux généré par un dispositif de projection (20) et une seconde lame de verre (12) formant une seconde face (02) opposée à la première face (01), caractérisé ce qu'il comprend les étapes suivantes : - laminage d'une feuille quart d'onde (18) sur un insert biseauté (16) de 5 manière à obtenir un ensemble insert - lame quart d'onde ; - assemblage de la première lame de verre (14) et de la seconde lame (12) de verre avec interposition dudit ensemble. 10. Procédé de fabrication selon la revendication 9, dans lequel l'étape 10 d'assemblage comprend les sous-étapes suivantes : - laminage dudit ensemble sur la première lame de verre (14) de manière à obtenir un nouvel ensemble ; - collage de la seconde lame de verre (12) sur le nouvel ensemble.".lower()

sents = sent_tokenize(claims, language="french")
patterns = []
for sent in sents:
    saos = get_SAO_fr(sent)
    if saos:
        patterns.extend(saos)

patterns = list(dict.fromkeys(patterns))
print(patterns)
print("\n")

model_inputs = []
for p in patterns:
    # TODO ajout d'article
    sent = " ".join([p[0], default_conjugator.conjugate(p[1]).conjug_info['Indicatif']['Présent']['3s'], p[2]])
    model_inputs.append(sent)
print(model_inputs)

