from dataset.PennFudan import PennFudanDataset

if __name__ == '__main__':
  test_dataset = PennFudanDataset("1_Object_detection_fine_tuning/data/PennFudanPed", None)
  test_dataset[90]