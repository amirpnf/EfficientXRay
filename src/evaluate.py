import torch
import matplotlib.pyplot as plt
from sklearn.metrics import RocCurveDisplay, roc_auc_score
from data_loader import get_labeled_loader
from classifier import XRayClassifier

def save_roc_curve():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = XRayClassifier(checkpoint_path="checkpoint_epoch_20.pth", num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load("final_xray_classifier.pth"))
    model.eval()
    
    test_loader = get_labeled_loader("data/labeled/test", batch_size=32, train=False)
    
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)[:, 1] 
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())
            

    plt.figure(figsize=(8, 6))
    RocCurveDisplay.from_predictions(all_labels, all_probs)
    
    plt.title(f"ROC Curve (AUC = {roc_auc_score(all_labels, all_probs):.2f})")
    plt.savefig("roc_curve.png")
    print("ROC curve saved as 'roc_curve.png'")

if __name__ == '__main__':
    save_roc_curve()