from concurrent.futures import thread
from get_SAO_fr import *
from pathlib import Path
import pandas as pd
import argparse, os, csv, sys, multiprocessing, psutil, pickle
from nltk import sent_tokenize
from tqdm import tqdm

csv.field_size_limit(sys.maxsize)

def read_tsv(file_path, labels_list, ipc_level=4):
    texts = []
    labels = []
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                texts.append(row['text'])
                labels_to_check = list(set([l[:ipc_level] for l in row['group_ids'].split(',')]))
                labels_checked = [l for l in labels_to_check if l in labels_list]
                labels.append(','.join(labels_checked))
        df = pd.DataFrame(zip(texts, labels), columns=['text','IPC' + str(ipc_level)])
    except KeyError:
        df = pd.read_csv(file_path, sep="\t", skipinitialspace=True, usecols=['text','group_ids'], dtype=object)
        df['group_ids'] = df['group_ids'].apply(str)
        df['text'] = df['text'].apply(str)
        df = df.rename(columns={"group_ids": 'IPC' + str(ipc_level)})
    return df

def remove_duplicate(seq):
    seen = set()
    seen_add = seen.add
    return [x for x in seq if not (x in seen or seen_add(x))]

def get_SAO_patterns(claims):
    res = []

    sents = sent_tokenize(claims, language="french")
    for sent in sents:
        saos = get_SAO_fr(sent)
        if saos:
            res.extend(saos)

    # remove duplications
    res = remove_duplicate(res)
    return res 

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", default='../data/INPI/new_extraction/output/inpi_new_final.csv', type=str, help="Path to input file of INPI-CLS data.")
    parser.add_argument("--in_dir", type=str, help="Path to input directory containing train.tsv and test.tsv")
    parser.add_argument("--out_dir", type=str, default='./results', help="Path to output directory to save models")
    parser.add_argument("--label_file", default='../data/ipc-sections/20210101/labels_group_id_4.tsv', type=str)
    parser.add_argument("--pred_level", default=4, type=int, choices={1, 3, 4, 6, 8}, help="Target IPC/CPC level of patent classification.")
    parser.add_argument("--split_by_year", default=2020, type=int, help="The year used to split data. (especially for INPI-CLS data <split_by_year as training and >=split_by_year as testing data).")
    parser.add_argument("--group_number", type=int, help="Manual multiprocessing?")

    parser.add_argument("--remove_determiner", action="store_true", help="Whether to remove determiners of subjects and subjects.")

    args = parser.parse_args()

    year = args.split_by_year
    label = 'IPC' + str(args.pred_level)

    if args.in_dir:
        data_name = args.in_dir.strip("/").split("/")[-1]
    else:
        data_name = "INPI"

    if args.remove_determiner:
        determiner_index = "noDeterminer"
    else:
        determiner_index = "withDeterminer"

    output_path = os.path.join(args.out_dir,'-'.join([data_name, determiner_index]))
    output_path = Path(output_path)

    if not output_path.exists():
        try:
            output_path.mkdir(parents=True)
            print(f"Created output directory {output_path}")
        except FileExistsError:
            print(f"{output_path} already exists!")

    print("***** Reading standard label file *****")
    if args.pred_level == 1:
        labels_list = ["A", "B", "C", "D", "E", "F", "G", "H"]
    else:
        with open(args.label_file, 'r') as in_f:
            lines = in_f.read().splitlines()
        labels_list = [l.split('\t')[0] for l in lines]

    print("***** Reading input data *****")
    if args.in_dir:                 
        df_train = read_tsv(os.path.join(args.in_dir, 'train.tsv'), labels_list, ipc_level=args.pred_level)
        df_test = read_tsv(os.path.join(args.in_dir, 'test.tsv'), labels_list, ipc_level=args.pred_level)
    else:
        df = pd.read_csv(args.in_file, dtype=str, engine="python")
        df.loc[:,'text'] = df['claims']

        df_train = df[df['date'].apply(lambda x: int(x[:4]) < year and int(x[:4])>=2000)]
        df_train = df_train[['text', label]].dropna()
        df_train[label] = df_train[label].apply(lambda x: ",".join([l for l in str(x).split(",") if l in labels_list]))
    
        df_test = df[df['date'].apply(lambda x: int(x[:4]) >= year)]
        df_test = df_test[['text', label]].dropna()
        df_test[label] = df_test[label].apply(lambda x: ",".join([l for l in str(x).split(",") if l in labels_list]))

    # df_train['SAO'] = df_train['text'].apply(get_SAO_patterns)
    # df_test['SAO'] = df_test['text'].apply(get_SAO_patterns)

    df_train.to_csv(os.path.join(output_path, "train.csv"), index=False)
    df_test.to_csv(os.path.join(output_path, "test.csv"), index=False)

    list_claims = df_train['text'].to_list() + df_test['text'].to_list()

    if args.group_number == 7:
        list_claims = list_claims[(len(list_claims)//8) * args.group_number:]
        print(f"Extraction of SAO patterns for claims from {(len(list_claims)//8) * args.group_number} to {len(list_claims)}.")
    else:
        list_claims = list_claims[(len(list_claims)//8) * args.group_number: (len(list_claims)//8) * (args.group_number+1)]
        print(f"Extraction of SAO patterns for claims from {(len(list_claims)//8) * args.group_number} to {(len(list_claims)//8 * (args.group_number+1))}.")

    res = []
    for claims in tqdm(list_claims):
        patterns = get_SAO_patterns(claims)
        res.append(patterns)

    with open(os.path.join(output_path, f"SAO_extracted_{str(args.group_number)}.pickle"), "wb") as out_f:
        pickle.dump(res, out_f, protocol=pickle.HIGHEST_PROTOCOL)

            


if __name__ == "__main__":
    main()
