# ToysAD-8K DS1 proposal and mask evaluation

The `kaggle-v3` folder contains the safe aggregate evidence from Notebook
06.1. The protected evaluator covered 100 scenes, 2,000 images, 7,700 reference
objects and 16,949 decoded SAM masks.

The reference-box SAM route passed its main gates. Recall at mask IoU 0.75 was
0.99234 and detection-aware mean mask IoU was 0.95267. This supports SAM 2.1
when the input box is reliable.

The automatic route reached 0.92481 recall and 0.89388 detection-aware mean
mask IoU. LocateAnything proposal precision at box IoU 0.50 was 0.78668 and
recall was 0.94494. These values missed the declared proposal gates, and the
automatic mask IoU also missed its 0.90 gate.

The recorded decision is `refine_or_replace_locateanything`. This is a valid
negative component result from the development screen, not an execution
failure. It does not change the accepted SAM result for reference boxes.

ClearML task `22494046a4ee4e6ba6e8eff8587fc650` stores the aggregate metrics,
tables, charts and run identity.

Per-object records, protected labels, masks and diagnostic image panels are
not stored in this repository.
