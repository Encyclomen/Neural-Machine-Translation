def load_dataset(dataset_class, src_corpus, trg_corpus, src_max_len=None, trg_max_len=None):

    dataset = dataset_class(src_corpus, trg_corpus, src_max_len, trg_max_len)
    # dataset_iter = torch.utils.data.DataLoader(dataset, batch_size, shuffle=shuffle, num_workers=num_workers,collate_fn=collate_fn)

    return dataset


def load_dataloader(dataset, dataloader_class, batch_size, **kwargs):
    dataloader = dataloader_class(dataset, batch_size, **kwargs)

    return dataloader