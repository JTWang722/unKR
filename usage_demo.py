import unKR
import pytorch_lightning as pl
import torch
args = unKR.setup_parser()
train_sampler = unKR.import_class('UKGUniSampler')(args)
test_sampler = unKR.import_class('UKGTestSampler')(args)
ukgdata = unKR.import_class('UKGDataMoudle')(args, train_sampler, test_sampler)
model = unKR.import_class('PASSLEAF')(args)
lit_model = unKR.import_class('PASSLEAFLitModel')(model, args)
trainer = pl.Trainer.from_argparse_args(args)
if not args.test_only:
    # Training
    trainer.fit(lit_model, datamodule=ukgdata)
    trainer.test(lit_model, datamodule=ukgdata)
else:
    # Test only
    lit_model.load_state_dict(torch.load(args.checkpoint_dir))
    trainer.test(lit_model, datamodule=ukgdata)
