import  torch
from    torchtext import data, datasets










def main():

    torch.manual_seed(123)

    TEXT = data.Field(tokenize='spacy')
    LABEL = data.LabelField(dtype=torch.float)
    train_data, test_data = datasets.IMDB.splits(TEXT, LABEL)
    print(len(train_data), len(test_data))








if __name__ == '__main__':
    main()