import spacy 

nlp = spacy.load('en_core_web_sm')
stopwords = nlp.Defaults.stop_words

with open("en_stopwords.txt") as inf:
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

############################ TEST ################################
from nltk import sent_tokenize
claims = "1. A display substrate, comprising: a base substrate; a thin film transistor on the base substrate; and a light shielding layer on the base substrate, the light shielding layer comprising a first light shielding layer and a second light shielding layer that are stacked, wherein an orthographic projection of an active layer of the thin film transistor on the base substrate is within an orthogonal projection of the light shielding layer on the base substrate, and the second light shielding layer comprises nanoparticles capable of absorbing light in a specific wavelength range.\n2. The display substrate according to claim 1, wherein a material of the first light shielding layer comprises monocrystalline silicon, polycrystalline silicon or amorphous silicon.\n3. The display substrate according to claims 1 or 2, wherein a material of the second light shielding layer further comprises silicon nitride or silicon carbide.\n4. The display substrate according to any one of claims 1-3, wherein a thickness of the first light shielding layer ranges from 400 Å to 600 Å, and a thickness of the second light shielding layer ranges from 200 Å to 500 Å.\n5. The display substrate according to any one of claims 1-4, wherein the nanoparticles are nano silicon particles.\n6. The display substrate according to claim 5, wherein particle sizes of the nano silicon particles range from 3nm to 5nm.\n7. The display substrate according to any one of claims 1-6, wherein the light is blue  light, and a wavelength of the blue light ranges from 420nm to 480 nm.\n8. The display substrate according to any one of claims 1-7, wherein the first light shielding layer is on a side of the second light shielding layer that is away from the base substrate; or the second light shielding layer is on a side of the first light shielding layer that is away from the base substrate\n9. The display substrate according to any one of claims 1-8, wherein the thin film transistor comprises a thin film transistor of a top gate structure or a thin film transistor of a bottom gate structure.\n10. The display substrate according to claim 9, wherein in a case where the thin film transistor has the bottom gate structure, the light shielding layer is on a side of the active layer that is away from the base substrate; or in a case where the thin film transistor has the top gate structure, the light shielding layer is disposed between the base substrate and the active layer.\n11. A display device, comprising the display substrate according to any one of claims 1-10.\n12. A manufacture method of a display substrate, comprising: providing a base substrate; forming a thin film transistor on the base substrate; and forming a light shielding layer, which comprises a first light shielding layer and a second light shielding layer, on the base substrate, wherein an orthographic projection of an active layer of the thin film transistor on the base substrate is within an orthogonal projection of the light shielding layer on the base substrate, and the second light shielding layer comprises nanoparticles capable of absorbing light in a specific wavelength range.\n13. The manufacture method of a display substrate according to claim 12, wherein a method of forming the second light shielding layer comprises spiral wave plasma chemical vapor deposition.\n14. The manufacture method of a display substrate according to claims 12 or 13, wherein the nanoparticles are nano silicon particles, and forming the second light shielding layer comprises: forming a second light shielding layer film comprising the nano silicon particles through a reaction gas comprising at least nitrogen, silane and hydrogen, or through a reaction gas comprising at least nitrogen, methane, silane and hydrogen; and performing a patterning process on the second light shielding layer film to form the second light shielding layer comprising the nano silicon particles.\n15. The manufacture method of a display substrate according to claim 14, wherein in a case where the second light shielding layer film is formed through the reaction gas comprising at least nitrogen, methane, silane and hydrogen, process conditions of the spiral wave plasma chemical vapor deposition comprise: a temperature ranging from 650 degrees Celsius to 750 degrees Celsius, power ranging from 400 Watts to 600 Watts, low pressure of pressure being up to 1.33 Pa, and a magnetic induction intensity ranging from 90 Gs to 130 Gs.\n16. The manufacture method of a display substrate according to claim 15, wherein the process conditions of the spiral wave plasma chemical vapor deposition comprise: the temperature being 700 degrees Celsius; the pressure is 1.33 Pa, the power being 500 watts; the magnetic induction intensity being 110 Gs and a volume ratio of the hydrogen, methane and silane being 1:2:40.\n17. The manufacture method of a display substrate according to any one of claims 12-16, wherein the light is blue light, and a wavelength of the blue light ranges from 420nm to 480  nm.\n18. The manufacture method of a display substrate according to any one of claims 12-16, wherein forming the light shielding layer, which comprises the first light shielding layer and the second light shielding layer, on the base substrate comprises: forming the first light shielding layer on the base substrate; and forming the second light shielding layer on the first light shielding layer.\n19. The manufacture method of a display substrate according to claim 18, wherein the light shielding layer is formed synchronously with the active layer in the thin film transistor, and the method comprises: after a thin film of the light shielding layer and a thin film of the active layer are sequentially formed on the base substrate, using a same mask for the thin film of the light shielding layer and the thin film of the active layer to form the light shielding layer and the active layer."

sents = sent_tokenize(claims, language="english")
for sent in sents:
    saos = get_SAO_en(sent)
    if saos:
        print(saos)