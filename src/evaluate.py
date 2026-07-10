import torch
from sklearn.metrics import confusion_matrix, classification_report
from data_loader import get_labeled_loader
from classifier import XRayClassifier

def evaluate_model():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = XRayClassifier(checkpoint_path="checkpoint_epoch_20.pth", num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("final_xray_classifier.pth"))
    model.eval()
    
    test_loader = get_labeled_loader("data/labeled/test", batch_size=32, train=False)
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=['NORMAL', 'PNEUMONIA']))
    
    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    evaluate_model()