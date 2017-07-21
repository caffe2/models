##Execute

```
python resnet50_trainer.py 
```

##Option

| option | default | description |
|:-----------|:------------|:------------|
| --train_data | None | Path to training data or 'everstore_sampler' |
| --test_data | None | Path to test data |
| --db_type | lmdb | Database type (such as lmdb or leveldb) |
| --gpus | 0? | Comma separated list of GPU devices to use |
| --num_gpus | 1 | "Number of GPU devices (instead of --gpus) |
| --num_channels | 3 | Number of color channels |
| --image_size | 227 | Input image size (to crop to) |
| --num_labels | 1000 | Number of labels |
| --batch_size | 32 | Batch size, total over all GPUs) |
| --epoch_size | 1500000 | Number of images/epoch, total over all machines |
| --num_epochs | 1000 | Num epochs |
| --base_learning_rate | 0.1 | Initial learning rate |
| --weight_decay | 1e-4 | Weight decay (L2 regularization) |
| --cudnn_workspace_limit_mb | 64 | CuDNN workspace limit in MBs |
| --num_shards | 1 | Number of machines(Nodes) in distributed run |
| --shard_id | 0 | Shard id |
| --run_id | ? | Num Unique run identifier (e.g. uuid) |
| --redis_host | ? | Host of Redis server (for rendezvous |
| --redis_port | 6379 | Port of Redis server (for rendezvous |
| --file_store_path | /tmp | Path to directory to use for rendezvous |
| --save_model_name | /resnet50_model | Save the trained model to a given name |
| --load_model_path | None | Load previously saved model to continue training |
| --use_cpu | False | Use CPU instead of GPU |

image_sizeに関して、Caffeのresnet50のモデルでは224となっている
https://github.com/rioyokotalab/caffe/blob/caffe-0.16/models/resnet50/train_val.prototxt
