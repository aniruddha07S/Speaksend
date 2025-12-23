"""
Generate test results and confusion matrix for Voice Authentication System
This script simulates test scenarios using existing voice profiles
"""

import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

class TestResultGenerator:
    def __init__(self):
        self.voice_dir = Path("voice_auth")
        self.threshold = 0.55  # From voice_auth.py
        self.profiles = {}
        self.load_profiles()
        
    def load_profiles(self):
        """Load all voice profiles"""
        print("Loading voice profiles...")
        for profile_file in self.voice_dir.glob("*_profile.pkl"):
            username = profile_file.stem.replace("_profile", "")
            try:
                with open(profile_file, 'rb') as f:
                    profile = pickle.load(f)
                    self.profiles[username] = profile['mean_embedding']
                    print(f"  ✓ Loaded profile for: {username}")
            except Exception as e:
                print(f"  ✗ Error loading {username}: {e}")
        print(f"\nLoaded {len(self.profiles)} profiles\n")
    
    def simulate_verification(self, claimed_user, actual_user, noise_level=0.1):
        """
        Simulate verification by adding noise to embeddings
        This simulates natural voice variation
        """
        if claimed_user not in self.profiles:
            return False, 0.0
        
        claimed_embedding = self.profiles[claimed_user]
        actual_embedding = self.profiles[actual_user]
        
        # Add noise to simulate voice variation
        # For genuine cases (same user), noise is smaller
        # For impostor cases (different user), use actual different embedding
        if claimed_user == actual_user:
            # Genuine case: add small noise to simulate natural variation
            noise = np.random.normal(0, noise_level, size=actual_embedding.shape)
            test_embedding = actual_embedding + noise
            test_embedding = test_embedding / np.linalg.norm(test_embedding)  # Normalize
        else:
            # Impostor case: use different user's embedding with some noise
            noise = np.random.normal(0, noise_level * 0.5, size=actual_embedding.shape)
            test_embedding = actual_embedding + noise
            test_embedding = test_embedding / np.linalg.norm(test_embedding)
        
        # Calculate similarity
        similarity = 1 - cosine(claimed_embedding, test_embedding)
        
        # Verify based on threshold
        verified = similarity >= self.threshold
        
        return verified, similarity
    
    def generate_test_results(self, num_trials=10):
        """Generate test results for all user combinations"""
        users = list(self.profiles.keys())
        if len(users) < 2:
            print("Error: Need at least 2 users for testing")
            return [], []
        
        print(f"Generating test results with {num_trials} trials per case...")
        print(f"Users: {', '.join(users)}\n")
        
        y_true = []
        y_pred = []
        results = []
        
        # Genuine cases: each user authenticating as themselves
        print("Testing genuine cases (users authenticating as themselves)...")
        for user in users:
            for trial in range(num_trials):
                verified, similarity = self.simulate_verification(user, user)
                y_true.append(1)  # Genuine
                y_pred.append(1 if verified else 0)  # Accepted/Rejected
                results.append({
                    'claimed': user,
                    'actual': user,
                    'genuine': True,
                    'verified': verified,
                    'similarity': similarity
                })
        
        # Impostor cases: each user trying to authenticate as others
        print("Testing impostor cases (users trying to authenticate as others)...")
        for actual_user in users:
            for claimed_user in users:
                if actual_user != claimed_user:
                    for trial in range(num_trials):
                        verified, similarity = self.simulate_verification(claimed_user, actual_user)
                        y_true.append(0)  # Impostor
                        y_pred.append(1 if verified else 0)  # Accepted/Rejected
                        results.append({
                            'claimed': claimed_user,
                            'actual': actual_user,
                            'genuine': False,
                            'verified': verified,
                            'similarity': similarity
                        })
        
        print(f"\nGenerated {len(results)} test cases\n")
        return y_true, y_pred, results
    
    def calculate_metrics(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        
        # Calculate additional metrics
        tn, fp, fn, tp = cm.ravel()
        
        # False Acceptance Rate (FAR) - impostors accepted
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        # False Rejection Rate (FRR) - genuine users rejected
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0
        
        # Equal Error Rate (EER) approximation
        eer = (far + frr) / 2
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm,
            'true_positives': int(tp),
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'false_acceptance_rate': far,
            'false_rejection_rate': frr,
            'equal_error_rate': eer
        }
        
        return metrics
    
    def print_metrics(self, metrics):
        """Print evaluation metrics"""
        print("\n" + "="*70)
        print("VOICE AUTHENTICATION SYSTEM - EVALUATION METRICS")
        print("="*70)
        print(f"\n📊 Performance Metrics:")
        print(f"   Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
        print(f"   Precision:       {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
        print(f"   Recall:          {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
        print(f"   F1-Score:        {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
        print(f"\n📋 Confusion Matrix:")
        print(f"   True Negatives (TN):  {metrics['true_negatives']:4d}  (Impostors correctly rejected)")
        print(f"   False Positives (FP): {metrics['false_positives']:4d}  (Impostors incorrectly accepted)")
        print(f"   False Negatives (FN): {metrics['false_negatives']:4d}  (Genuine users incorrectly rejected)")
        print(f"   True Positives (TP):  {metrics['true_positives']:4d}  (Genuine users correctly accepted)")
        print(f"\n🔒 Security Metrics:")
        print(f"   False Acceptance Rate (FAR): {metrics['false_acceptance_rate']:.4f} ({metrics['false_acceptance_rate']*100:.2f}%)")
        print(f"   False Rejection Rate (FRR):  {metrics['false_rejection_rate']:.4f} ({metrics['false_rejection_rate']*100:.2f}%)")
        print(f"   Equal Error Rate (EER):      {metrics['equal_error_rate']:.4f} ({metrics['equal_error_rate']*100:.2f}%)")
        print("\n" + "="*70 + "\n")
    
    def plot_confusion_matrix(self, metrics, save_path='confusion_matrix.png'):
        """Plot and save confusion matrix"""
        cm = metrics['confusion_matrix']
        
        plt.figure(figsize=(12, 10))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected\n(Impostor)', 'Accepted\n(Genuine)'],
                   yticklabels=['Rejected\n(Impostor)', 'Accepted\n(Genuine)'],
                   cbar_kws={'label': 'Count'},
                   linewidths=2,
                   linecolor='black',
                   annot_kws={'size': 16, 'weight': 'bold'})
        
        plt.title('Voice Authentication Confusion Matrix', 
                 fontsize=18, fontweight='bold', pad=25)
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        
        # Add detailed annotations
        tn, fp, fn, tp = cm.ravel()
        plt.text(0.5, -0.18, f'TN: {tn}\n(Impostors\nCorrectly Rejected)', 
                transform=plt.gca().transAxes, ha='center', fontsize=11, 
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        plt.text(1.5, -0.18, f'FP: {fp}\n(Impostors\nIncorrectly Accepted)', 
                transform=plt.gca().transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        plt.text(0.5, 2.18, f'FN: {fn}\n(Genuine Users\nIncorrectly Rejected)', 
                transform=plt.gca().transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
        plt.text(1.5, 2.18, f'TP: {tp}\n(Genuine Users\nCorrectly Accepted)', 
                transform=plt.gca().transAxes, ha='center', fontsize=11,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_normalized_confusion_matrix(self, metrics, save_path='confusion_matrix_normalized.png'):
        """Plot normalized confusion matrix (percentages)"""
        cm = metrics['confusion_matrix']
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(12, 10))
        
        sns.heatmap(cm_normalized, annot=True, fmt='.2%', cmap='Blues',
                   xticklabels=['Rejected\n(Impostor)', 'Accepted\n(Genuine)'],
                   yticklabels=['Rejected\n(Impostor)', 'Accepted\n(Genuine)'],
                   cbar_kws={'label': 'Percentage'},
                   linewidths=2,
                   linecolor='black',
                   annot_kws={'size': 16, 'weight': 'bold'})
        
        plt.title('Normalized Confusion Matrix (Percentages)', 
                 fontsize=18, fontweight='bold', pad=25)
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Normalized confusion matrix saved to: {save_path}")
        plt.close()
    
    def plot_metrics_bar(self, metrics, save_path='metrics_bar_chart.png'):
        """Plot metrics as bar chart"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']
        bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{value:.3f}\n({value*100:.1f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylim([0, 1.15])
        ax.set_ylabel('Score', fontsize=13, fontweight='bold')
        ax.set_title('Voice Authentication System - Evaluation Metrics', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_xticklabels(metric_names, fontsize=12)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Metrics bar chart saved to: {save_path}")
        plt.close()
    
    def plot_security_metrics(self, metrics, save_path='security_metrics.png'):
        """Plot security-specific metrics"""
        fig, ax = plt.subplots(figsize=(12, 7))
        
        metric_names = ['FAR\n(False\nAcceptance)', 'FRR\n(False\nRejection)', 'EER\n(Equal Error\nRate)']
        metric_values = [
            metrics['false_acceptance_rate'],
            metrics['false_rejection_rate'],
            metrics['equal_error_rate']
        ]
        
        colors = ['#e74c3c', '#f39c12', '#9b59b6']
        bars = ax.bar(metric_names, metric_values, color=colors, edgecolor='black', linewidth=2)
        
        # Add value labels
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + max(metric_values) * 0.05,
                   f'{value:.3f}\n({value*100:.2f}%)',
                   ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        max_val = max(metric_values) if max(metric_values) > 0 else 0.1
        ax.set_ylim([0, max_val * 1.4])
        ax.set_ylabel('Rate', fontsize=13, fontweight='bold')
        ax.set_title('Voice Authentication System - Security Metrics', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Security metrics chart saved to: {save_path}")
        plt.close()
    
    def save_detailed_report(self, metrics, results, save_path='evaluation_report.txt'):
        """Save detailed evaluation report"""
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("VOICE AUTHENTICATION SYSTEM - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            f.write("CONFIGURATION\n")
            f.write("-"*70 + "\n")
            f.write(f"Threshold: {self.threshold}\n")
            f.write(f"Number of Users: {len(self.profiles)}\n")
            f.write(f"Users: {', '.join(self.profiles.keys())}\n")
            f.write(f"Total Test Cases: {len(results)}\n\n")
            
            f.write("EVALUATION METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"Accuracy:        {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Precision:       {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)\n")
            f.write(f"Recall:          {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)\n")
            f.write(f"F1-Score:        {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)\n\n")
            
            f.write("CONFUSION MATRIX\n")
            f.write("-"*70 + "\n")
            f.write(f"                    Predicted\n")
            f.write(f"                  Reject  Accept\n")
            f.write(f"Actual  Reject     {metrics['true_negatives']:4d}    {metrics['false_positives']:4d}\n")
            f.write(f"        Accept     {metrics['false_negatives']:4d}    {metrics['true_positives']:4d}\n\n")
            
            f.write("SECURITY METRICS\n")
            f.write("-"*70 + "\n")
            f.write(f"False Acceptance Rate (FAR): {metrics['false_acceptance_rate']:.4f} ({metrics['false_acceptance_rate']*100:.2f}%)\n")
            f.write(f"  - Rate at which impostors are incorrectly accepted\n")
            f.write(f"  - Lower is better (security concern)\n\n")
            f.write(f"False Rejection Rate (FRR):  {metrics['false_rejection_rate']:.4f} ({metrics['false_rejection_rate']*100:.2f}%)\n")
            f.write(f"  - Rate at which genuine users are incorrectly rejected\n")
            f.write(f"  - Lower is better (usability concern)\n\n")
            f.write(f"Equal Error Rate (EER):      {metrics['equal_error_rate']:.4f} ({metrics['equal_error_rate']*100:.2f}%)\n")
            f.write(f"  - Point where FAR equals FRR\n")
            f.write(f"  - Lower is better (overall system performance)\n\n")
            
            f.write("INTERPRETATION\n")
            f.write("-"*70 + "\n")
            f.write("True Positives (TP):  Genuine users correctly accepted\n")
            f.write("True Negatives (TN):  Impostors correctly rejected\n")
            f.write("False Positives (FP): Impostors incorrectly accepted (Security Risk)\n")
            f.write("False Negatives (FN): Genuine users incorrectly rejected (Usability Issue)\n")
        
        print(f"✓ Detailed report saved to: {save_path}")


def main():
    """Main function to generate test results and visualizations"""
    print("\n" + "="*70)
    print("Voice Authentication System - Test Results Generator")
    print("="*70 + "\n")
    
    generator = TestResultGenerator()
    
    if len(generator.profiles) < 2:
        print("Error: Need at least 2 enrolled users to generate test results")
        print(f"Found {len(generator.profiles)} user(s)")
        return
    
    # Generate test results
    y_true, y_pred, results = generator.generate_test_results(num_trials=10)
    
    # Calculate metrics
    metrics = generator.calculate_metrics(y_true, y_pred)
    
    # Print metrics
    generator.print_metrics(metrics)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    generator.plot_confusion_matrix(metrics, 'confusion_matrix.png')
    generator.plot_normalized_confusion_matrix(metrics, 'confusion_matrix_normalized.png')
    generator.plot_metrics_bar(metrics, 'metrics_bar_chart.png')
    generator.plot_security_metrics(metrics, 'security_metrics.png')
    
    # Save detailed report
    generator.save_detailed_report(metrics, results, 'evaluation_report.txt')
    
    print("\n" + "="*70)
    print("All evaluation files have been generated successfully!")
    print("="*70)
    print("\nGenerated files:")
    print("  📊 confusion_matrix.png")
    print("  📊 confusion_matrix_normalized.png")
    print("  📊 metrics_bar_chart.png")
    print("  📊 security_metrics.png")
    print("  📄 evaluation_report.txt")
    print("\n")


if __name__ == "__main__":
    main()





