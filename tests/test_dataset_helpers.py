from selfattention_audio import dataset_helpers as dhelp

def test_AudioBrainDataset(full_data) -> None:
    datasets = [dhelp.AudioBrainDataset(*data) for data in full_data]
    for ds, dt in zip(datasets, full_data):
        assert len(ds) == dt[0].shape[0]
        assert (ds[10][0] == dt[0][10]).all()
