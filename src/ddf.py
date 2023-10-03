from datetime import datetime
from deepreg.predict import predict
from deepreg.train import train


def train_ddf():

    log_dir = "/home/alicia/data_personal/regnet_ckpt"
    # directory to save the training checkpoints
    euler_version = 0
    exp_name = "logs_train/" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
                f"_eulerGPU_ddf_v{euler_version}_epoch20000"
    config_path = [f"configs/config_eulerGPU_ddf_v{euler_version}.yaml"]

    ckpt_name = f'20230928-020441_eulerGPU_ddf_v{euler_version}_epoch2000'
    ckpt_num = 2000
    ckpt_path = \
        f'/home/alicia/data_personal/regnet_ckpt/logs_train/{ckpt_name}/save/ckpt-{ckpt_num}'
    # train
    train(
        gpu="0",
        config_path=config_path,
        gpu_allow_growth=True,
        ckpt_path=ckpt_path,
        log_dir=log_dir,
        exp_name=exp_name,
    )


def test_ddf():

    ckpt_num = 2000
    log_dir = "/home/alicia/data_personal/regnet_ckpt"
    exp_name = "logs_predict/" + datetime.now().strftime("%Y%m%d-%H%M%S") + \
                f"_euler_ddf_epoch2000_ckpt{ckpt_num}_test-on-valid"
    ckpt_name = "20230815-174102_euler_ddf_epoch2000"
    ckpt_path = \
        f"/home/alicia/data_personal/regnet_ckpt/logs_train/{ckpt_name}/save/ckpt-{ckpt_num}"
    #config_path = ["configs/config_euler_ddf.yaml"]
    config_path = ['configs/config_euler_ddf_test-on-valid.yaml']

    predict(
        gpu="4",
        gpu_allow_growth=True,
        ckpt_path=ckpt_path,
        split="test",
        batch_size=4,
        log_dir=log_dir,
        exp_name=exp_name,
        config_path=config_path,
    )


if __name__ == "__main__":
    train_ddf()
