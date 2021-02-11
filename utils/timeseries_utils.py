from data import amsterdam


def prepare_amsterdam(amsterdam_loader, settings):
    assert amsterdam_loader.pad_before == False
    raw_data, padding_mask, (train_idx, val_idx, test_idx) = \
        amsterdam_loader.load_reshape_split_data(force_refresh=settings["data_loading_force_refresh"])
    processed_data, imputed_processed_data = amsterdam.preprocess_data(
        raw_data, 
        padding_mask, 
        padding_fill=settings["pad_val"],
        time_feature_included=settings["include_time"],
    )
    seq_lens = amsterdam.padding_mask_to_seq_lens(padding_mask)
    return imputed_processed_data, seq_lens, (train_idx, val_idx, test_idx)
