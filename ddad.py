from asyncio import constants
from typing import Any
import torch
from unet import *
from dataset import *
from visualize import *
from anomaly_map import *
from metrics import *
from feature_extractor import *
from reconstruction import *
import pandas as pd #TODO:是否导入了对应的pandas库呢？记得与pytorch版本看齐
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2"

class DDAD:
    def __init__(self, unet, config) -> None:
        self.test_dataset = Dataset_maker(
            root= config.data.data_dir,
            category=config.data.category,
            config = config,
            is_train=False,
        )
        self.testloader = torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size= config.data.test_batch_size,
            shuffle=False,
            # num_workers= config.model.num_workers,
            num_workers= 0,
            drop_last=False,
        )
        self.unet = unet
        self.config = config
        self.reconstruction = Reconstruction(self.unet, self.config)
        # self.transform = transforms.Compose([
        #                     transforms.CenterCrop((224)), 
        #                 ])

    def __call__(self) -> Any:
        feature_extractor = domain_adaptation(self.unet, self.config, fine_tune=False)
        feature_extractor.eval()
        
        labels_list = []
        predictions= []
        image_paths = [] # 存储图片路径
        # anomaly_map_list = []
        # gt_list = []
        # reconstructed_list = []
        # forward_list = []


        with torch.no_grad():
            for input, image_file in self.testloader:
                input = input.to(self.config.model.device)
                x0 = self.reconstruction(input, input, self.config.model.w)[-1]
                anomaly_map = heat_map(x0, input, feature_extractor, self.config)
                for pred, path in zip(anomaly_map, image_file):
                    image_paths.append(path)
                    predictions.append(torch.max(pred).item())



                # 记录图片路径
                # batch_paths = [p for p in self.test_dataset.image_files]
                # image_paths.extend(batch_paths)
                # anomaly_map = self.transform(anomaly_map)
                # gt = self.transform(gt)

                # forward_list.append(input)
                # anomaly_map_list.append(anomaly_map)


                # gt_list.append(gt)
                # reconstructed_list.append(x0)

                # for pred, label in zip(anomaly_map, labels):
                #     labels_list.append(0 if label == 'good' else 1)
                #     predictions.append(torch.max(pred).item())

        
        # metric = Metric(labels_list, predictions, None, None, self.config)
        # binary_predictions = metric.get_predictions()
        
         # 自动阈值或固定阈值
        threshold = np.percentile(predictions, 95)  # 可选：用95分位作为异常阈值
        binary_predictions = [1 if p > threshold else 0 for p in predictions]


        # 保存结果到CSV
        results_df = pd.DataFrame({
            'image_path': image_paths,
            'prediction': binary_predictions 
        })
        results_df.to_csv('test_results.csv', index=False)

        # metric.optimal_threshold()
        # if self.config.metrics.auroc:
        #     print('AUROC: ({:.1f},{:.1f})'.format(metric.image_auroc() * 100, metric.pixel_auroc() * 100))
        # if self.config.metrics.pro:
        #     print('PRO: {:.1f}'.format(metric.pixel_pro() * 100))
        # if self.config.metrics.misclassifications:
        #     metric.miscalssified()
        # reconstructed_list = torch.cat(reconstructed_list, dim=0)
        # forward_list = torch.cat(forward_list, dim=0)
        # anomaly_map_list = torch.cat(anomaly_map_list, dim=0)
        # pred_mask = (anomaly_map_list > metric.threshold).float()
        # gt_list = torch.cat(gt_list, dim=0)
        # if not os.path.exists('results'):
        #         os.mkdir('results')
        # if self.config.metrics.visualisation:
        #     visualize(forward_list, reconstructed_list, gt_list, pred_mask, anomaly_map_list, self.config.data.category)
