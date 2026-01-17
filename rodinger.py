"""
Exercise List 2: Explainability and Uncertainty
Student ID: ai24m033
Random seed: 33

This file implements all 5 exercises demonstrating various explainability techniques:
1. Permutation Importance & Partial Dependence
2. Integrated Gradients (Model-Specific)
3. LIME for Images (Model-Agnostic)
4. LIME for Text (Sentiment Analysis)
5. Uncertainty and Robustness
"""

import os
import sys
import warnings
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
SEED = 33
np.random.seed(SEED)
random.seed(SEED)

# Add the explainability data and models directory to path
MODELS_DIR = os.path.join(os.path.dirname(__file__), "explainability data and models")
sys.path.insert(0, MODELS_DIR)

# ============================================================================
# EXERCISE 1: Permutation Importance & Partial Dependence
# ============================================================================

def exercise_1_permutation_importance():
    """
    Permutation Importance: Measures feature importance by shuffling
    each feature and measuring performance drop.

    Partial Dependence: Shows how predictions change as we vary one feature.
    """
    print("=" * 70)
    print("EXERCISE 1: Permutation Importance & Partial Dependence")
    print("=" * 70)

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, recall_score
    from sklearn.inspection import permutation_importance, PartialDependenceDisplay

    # Load data
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), "german_credit_reduced.csv"))
    print(f"Loaded German Credit data: {df.shape[0]} samples, {df.shape[1]} features")

    # Prepare features and target
    X = df.drop('risky', axis=1)
    y = df['risky']
    feature_names = X.columns.tolist()

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=SEED, stratify=y
    )

    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    print(f"Class distribution: {y.value_counts().to_dict()}")

    # Train models with different configurations
    models = {
        'Unweighted': DecisionTreeClassifier(random_state=SEED, max_depth=5),
        'Class-Weighted': DecisionTreeClassifier(
            random_state=SEED, max_depth=5, class_weight='balanced'
        ),
        'Dual-Weighted': DecisionTreeClassifier(
            random_state=SEED, max_depth=5, class_weight={0: 1, 1: 5}  # Higher weight for risky class
        )
    }

    print("\n" + "-" * 50)
    print("Training Decision Tree Models...")
    print("-" * 50)

    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        tpr = recall_score(y_test, y_pred)  # TPR = recall for positive class
        print(f"\n{name} Model:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  TPR (Recall for risky class): {tpr:.3f}")

    # Permutation importance with different metrics
    print("\n" + "-" * 50)
    print("Computing Permutation Importance...")
    print("-" * 50)

    # Focus on Dual-Weighted model
    model = trained_models['Dual-Weighted']

    # Permutation importance with ACCURACY
    print("\n1. Permutation Importance using ACCURACY:")
    perm_imp_acc = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=SEED, scoring='accuracy'
    )

    # Sort by importance
    sorted_idx_acc = perm_imp_acc.importances_mean.argsort()[::-1]
    print("\nFeature importances (sorted by mean):")
    for idx in sorted_idx_acc:
        print(f"  {feature_names[idx]:20s}: {perm_imp_acc.importances_mean[idx]:.4f} "
              f"(+/- {perm_imp_acc.importances_std[idx]:.4f})")

    # Permutation importance with TPR (recall)
    print("\n2. Permutation Importance using TPR (True Positive Rate):")
    perm_imp_tpr = permutation_importance(
        model, X_test, y_test, n_repeats=30, random_state=SEED, scoring='recall'
    )

    sorted_idx_tpr = perm_imp_tpr.importances_mean.argsort()[::-1]
    print("\nFeature importances (sorted by mean):")
    for idx in sorted_idx_tpr:
        print(f"  {feature_names[idx]:20s}: {perm_imp_tpr.importances_mean[idx]:.4f} "
              f"(+/- {perm_imp_tpr.importances_std[idx]:.4f})")

    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy-based importance
    ax1 = axes[0]
    ax1.barh(range(len(feature_names)),
             [perm_imp_acc.importances_mean[i] for i in sorted_idx_acc],
             xerr=[perm_imp_acc.importances_std[i] for i in sorted_idx_acc],
             color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(feature_names)))
    ax1.set_yticklabels([feature_names[i] for i in sorted_idx_acc])
    ax1.set_xlabel('Mean Importance')
    ax1.set_title('Permutation Importance (ACCURACY)')
    ax1.invert_yaxis()

    # TPR-based importance
    ax2 = axes[1]
    ax2.barh(range(len(feature_names)),
             [perm_imp_tpr.importances_mean[i] for i in sorted_idx_tpr],
             xerr=[perm_imp_tpr.importances_std[i] for i in sorted_idx_tpr],
             color='coral', alpha=0.7)
    ax2.set_yticks(range(len(feature_names)))
    ax2.set_yticklabels([feature_names[i] for i in sorted_idx_tpr])
    ax2.set_xlabel('Mean Importance')
    ax2.set_title('Permutation Importance (TPR)')
    ax2.invert_yaxis()

    plt.suptitle('Permutation Importance Comparison: Accuracy vs TPR', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex1_permutation_importance.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Partial Dependence Plots
    print("\n" + "-" * 50)
    print("Generating Partial Dependence Plots...")
    print("-" * 50)

    # Get top 4 features based on accuracy importance
    top_features = [feature_names[i] for i in sorted_idx_acc[:4]]
    print(f"\nTop 4 features for PDP: {top_features}")

    fig, ax = plt.subplots(figsize=(12, 8))
    PartialDependenceDisplay.from_estimator(
        model, X_test, top_features,
        kind='both',  # Shows both ICE and PDP
        ax=ax,
        random_state=SEED
    )
    plt.suptitle('Partial Dependence Plots (Dual-Weighted Decision Tree)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig('ex1_partial_dependence.png', dpi=150, bbox_inches='tight')
    plt.show()

    return trained_models


# ============================================================================
# EXERCISE 2: Integrated Gradients
# ============================================================================

def exercise_2_integrated_gradients():
    """
    Integrated Gradients: Model-specific explainability using gradients.
    Shows which pixels contribute most to the prediction by integrating
    gradients along a path from baseline to input.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 2: Integrated Gradients (Model-Specific)")
    print("=" * 70)

    import torch
    import torch.nn.functional as F
    from torchvision import transforms

    # Import from the models directory
    from model import ConvResNet
    from dataset import WaterbirdDataset

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load the pre-trained model
    checkpoint_path = os.path.join(MODELS_DIR, "checkpoints", "model_resnet18.pt")
    model = ConvResNet(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Loaded pre-trained ResNet18 bird classifier")

    # Load dataset
    annotation_file = os.path.join(MODELS_DIR, "data", "annotation.csv")
    image_dir = os.path.join(MODELS_DIR, "data", "CUB_200_2011", "images")

    # Use model's transforms
    transform = model.transforms
    dataset = WaterbirdDataset(annotation_file, image_dir, transform=transform, train=False)
    label_names = dataset.label_names
    print(f"Loaded test dataset: {len(dataset)} images")
    print(f"Labels: {label_names}")

    def calculate_integrated_gradients(model, input_tensor, target_class, baseline=None, num_steps=50):
        """Calculate integrated gradients for a given input."""
        if baseline is None:
            baseline = torch.zeros_like(input_tensor)

        # Create interpolated inputs
        alphas = torch.linspace(0, 1, num_steps + 1).to(input_tensor.device)
        scaled_inputs = torch.stack([
            baseline + alpha * (input_tensor - baseline) for alpha in alphas
        ])

        # Enable gradients
        scaled_inputs.requires_grad_(True)

        # Forward pass
        outputs = model(scaled_inputs)
        target_outputs = outputs[:, target_class]

        # Backward pass
        target_outputs.sum().backward()

        # Compute integrated gradients
        gradients = scaled_inputs.grad
        avg_gradients = gradients.mean(dim=0)
        integrated_grads = (input_tensor - baseline) * avg_gradients

        return integrated_grads

    def visualize_ig(original_img, ig_attr, predicted_label, true_label, title, quantile=0.04):
        """Visualize integrated gradients as heatmap."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Original image
        axes[0].imshow(original_img)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        # Integrated gradients heatmap
        ig_np = ig_attr.detach().cpu().numpy().transpose(1, 2, 0)
        ig_sum = np.sum(ig_np, axis=2)

        # Threshold to show only top attributions
        thresh_min, thresh_max = np.quantile(ig_sum, [quantile/2, 1 - quantile/2])
        ig_sum[np.logical_and(ig_sum > thresh_min, ig_sum < thresh_max)] = 0
        ig_sum = np.abs(ig_sum)

        im = axes[1].imshow(ig_sum, cmap='Blues')
        axes[1].set_title('Integrated Gradients')
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1], fraction=0.046)

        # Overlay
        axes[2].imshow(original_img)
        axes[2].imshow(ig_sum, cmap='jet', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        pred_name = label_names[predicted_label]
        true_name = label_names[true_label]
        correct = "CORRECT" if predicted_label == true_label else "MISCLASSIFIED"
        plt.suptitle(f'{title}\nPredicted: {pred_name}, True: {true_name} [{correct}]', fontsize=12)
        plt.tight_layout()
        return fig

    # Find correctly classified and misclassified examples
    print("\n" + "-" * 50)
    print("Finding examples for analysis...")
    print("-" * 50)

    correct_waterbird = None
    correct_landbird = None
    misclassified_waterbird = None
    misclassified_landbird = None

    # We need raw images for visualization
    raw_dataset = WaterbirdDataset(annotation_file, image_dir, transform=transforms.ToTensor(), train=False)

    # Sample images to find examples
    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset))[:200].tolist()

    for idx in indices:
        img_tensor, true_label = dataset[idx]
        raw_img_tensor, _ = raw_dataset[idx]

        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            pred_label = output.argmax(dim=1).item()

        # Store examples
        if pred_label == true_label == 0 and correct_waterbird is None:  # Correct waterbird
            correct_waterbird = (idx, img_tensor, raw_img_tensor, true_label, pred_label)
        elif pred_label == true_label == 1 and correct_landbird is None:  # Correct landbird
            correct_landbird = (idx, img_tensor, raw_img_tensor, true_label, pred_label)
        elif pred_label != true_label and true_label == 0 and misclassified_waterbird is None:
            misclassified_waterbird = (idx, img_tensor, raw_img_tensor, true_label, pred_label)
        elif pred_label != true_label and true_label == 1 and misclassified_landbird is None:
            misclassified_landbird = (idx, img_tensor, raw_img_tensor, true_label, pred_label)

        if all([correct_waterbird, correct_landbird, misclassified_waterbird, misclassified_landbird]):
            break

    examples = {
        'Correct Waterbird': correct_waterbird,
        'Correct Landbird': correct_landbird,
        'Misclassified Waterbird': misclassified_waterbird,
        'Misclassified Landbird': misclassified_landbird
    }

    # Process each example
    print("\n" + "-" * 50)
    print("Computing Integrated Gradients for each example...")
    print("-" * 50)

    figs = []
    for name, example in examples.items():
        if example is None:
            print(f"\n{name}: No example found")
            continue

        idx, img_tensor, raw_img_tensor, true_label, pred_label = example
        print(f"\n{name} (idx={idx}):")
        print(f"  True label: {label_names[true_label]}")
        print(f"  Predicted: {label_names[pred_label]}")

        # Move to device and compute IG
        img_tensor = img_tensor.to(device)

        # Use predicted class for attribution (what the model saw)
        ig = calculate_integrated_gradients(model, img_tensor, pred_label)

        # Convert raw image for visualization
        raw_img = raw_img_tensor.permute(1, 2, 0).numpy()
        raw_img = (raw_img * 255).astype(np.uint8)
        raw_img_pil = Image.fromarray(raw_img).resize((224, 224))

        fig = visualize_ig(raw_img_pil, ig, pred_label, true_label, name)
        figs.append(fig)
        plt.savefig(f'ex2_ig_{name.lower().replace(" ", "_")}.png', dpi=150, bbox_inches='tight')
        plt.show()


# ============================================================================
# EXERCISE 3: LIME for Images
# ============================================================================

def exercise_3_lime_images():
    """
    LIME (Local Interpretable Model-agnostic Explanations) for images.
    Explains predictions by fitting a simple linear model on superpixel perturbations.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 3: LIME for Images (Model-Agnostic)")
    print("=" * 70)

    import torch
    from torchvision import transforms
    from lime import lime_image
    from skimage.segmentation import mark_boundaries

    from model import ConvResNet
    from dataset import WaterbirdDataset

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = os.path.join(MODELS_DIR, "checkpoints", "model_resnet18.pt")
    model = ConvResNet(num_classes=2)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    print("Loaded pre-trained ResNet18 bird classifier")

    # Load dataset
    annotation_file = os.path.join(MODELS_DIR, "data", "annotation.csv")
    image_dir = os.path.join(MODELS_DIR, "data", "CUB_200_2011", "images")

    transform = model.transforms
    dataset = WaterbirdDataset(annotation_file, image_dir, transform=transform, train=False)
    label_names = dataset.label_names

    # Raw images for LIME
    raw_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    raw_dataset = WaterbirdDataset(annotation_file, image_dir, transform=raw_transform, train=False)

    print(f"Loaded test dataset: {len(dataset)} images")

    def predict_fn(images):
        """Prediction function for LIME - expects numpy array of shape (N, H, W, 3)"""
        # Convert numpy to tensor
        batch = torch.stack([
            transform(Image.fromarray((img * 255).astype(np.uint8)))
            for img in images
        ]).to(device)

        with torch.no_grad():
            outputs = model(batch)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()

    # Initialize LIME explainer
    explainer = lime_image.LimeImageExplainer()

    # Find examples (similar to Exercise 2)
    print("\n" + "-" * 50)
    print("Finding examples for LIME analysis...")
    print("-" * 50)

    examples = {'correct_waterbird': None, 'correct_landbird': None,
                'misclassified_waterbird': None, 'misclassified_landbird': None}

    torch.manual_seed(SEED)
    indices = torch.randperm(len(dataset))[:200].tolist()

    for idx in indices:
        img_tensor, true_label = dataset[idx]
        raw_img_tensor, _ = raw_dataset[idx]

        with torch.no_grad():
            output = model(img_tensor.unsqueeze(0).to(device))
            pred_label = output.argmax(dim=1).item()

        if pred_label == true_label == 0 and examples['correct_waterbird'] is None:
            examples['correct_waterbird'] = (idx, raw_img_tensor, true_label, pred_label)
        elif pred_label == true_label == 1 and examples['correct_landbird'] is None:
            examples['correct_landbird'] = (idx, raw_img_tensor, true_label, pred_label)
        elif pred_label != true_label and true_label == 0 and examples['misclassified_waterbird'] is None:
            examples['misclassified_waterbird'] = (idx, raw_img_tensor, true_label, pred_label)
        elif pred_label != true_label and true_label == 1 and examples['misclassified_landbird'] is None:
            examples['misclassified_landbird'] = (idx, raw_img_tensor, true_label, pred_label)

        if all(v is not None for v in examples.values()):
            break

    # Process each example with LIME
    print("\n" + "-" * 50)
    print("Computing LIME explanations (this may take a moment)...")
    print("-" * 50)

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))

    for i, (name, example) in enumerate(examples.items()):
        if example is None:
            print(f"\n{name}: No example found")
            continue

        idx, raw_img_tensor, true_label, pred_label = example
        print(f"\n{name} (idx={idx}):")
        print(f"  True: {label_names[true_label]}, Predicted: {label_names[pred_label]}")

        # Convert to numpy for LIME
        img_np = raw_img_tensor.permute(1, 2, 0).numpy()

        # Run LIME
        explanation = explainer.explain_instance(
            img_np,
            predict_fn,
            top_labels=2,
            hide_color=0,
            num_samples=500,  # Reduced for speed
            random_seed=SEED
        )

        # Get explanation for predicted class
        temp, mask = explanation.get_image_and_mask(
            pred_label,
            positive_only=False,
            num_features=10,
            hide_rest=False
        )

        # Plot original
        row = i // 2
        col = (i % 2) * 2

        axes[row, col].imshow(img_np)
        axes[row, col].set_title(f'{name}\nTrue: {label_names[true_label]}')
        axes[row, col].axis('off')

        # Plot LIME explanation
        img_boundary = mark_boundaries(temp, mask, color=(1, 1, 0), mode='thick')
        axes[row, col + 1].imshow(img_boundary)
        axes[row, col + 1].set_title(f'LIME: Pred={label_names[pred_label]}')
        axes[row, col + 1].axis('off')

    plt.suptitle('LIME Explanations for Bird Classification', fontsize=14)
    plt.tight_layout()
    plt.savefig('ex3_lime_images.png', dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# EXERCISE 4: LIME for Text
# ============================================================================

def exercise_4_lime_text():
    """
    LIME for text sentiment analysis.
    Shows which words most influence sentiment predictions.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 4: LIME for Text (Sentiment Analysis)")
    print("=" * 70)

    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    from transformers import pipeline
    from lime.lime_text import LimeTextExplainer
    import gc

    # Load reviews data
    reviews_path = os.path.join(os.path.dirname(__file__), "reviews.csv")
    df = pd.read_csv(reviews_path)

    # Sample 2000 reviews with seed 33
    subset_df = df.sample(n=2000, random_state=SEED).reset_index(drop=True)
    print(f"Sampled {len(subset_df)} reviews with seed {SEED}")

    # Basic EDA
    print("\n" + "-" * 50)
    print("Exploratory Data Analysis")
    print("-" * 50)

    # Polarity distribution (1=negative, 2=positive based on data inspection)
    print(f"\nPolarity distribution:")
    print(subset_df['polarity'].value_counts().to_string())

    # Text length statistics
    subset_df['text_length'] = subset_df['text'].str.len()
    subset_df['word_count'] = subset_df['text'].str.split().str.len()

    print(f"\nText length statistics:")
    print(f"  Mean characters: {subset_df['text_length'].mean():.1f}")
    print(f"  Median characters: {subset_df['text_length'].median():.1f}")
    print(f"  Mean words: {subset_df['word_count'].mean():.1f}")
    print(f"  Median words: {subset_df['word_count'].median():.1f}")

    # Load sentiment pipeline
    print("\nLoading DistilBERT sentiment model...")
    import torch

    if torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS (Apple Silicon GPU)")
    else:
        device = "cpu"
        print("Using CPU")

    sentiment_pipeline = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
        device=device
    )

    # Run sentiment analysis one at a time to avoid memory issues
    print("Running sentiment analysis on all reviews...")
    texts = subset_df['text'].tolist()

    results = []
    for i, text in enumerate(texts):
        result = sentiment_pipeline(text[:512])[0]
        results.append(result)
        if (i + 1) % 500 == 0:
            print(f"  Processed {i + 1}/{len(texts)} reviews...")
            gc.collect()

    subset_df['sentiment'] = [r['label'] for r in results]
    subset_df['confidence'] = [r['score'] for r in results]

    print(f"\nModel predictions:")
    print(subset_df['sentiment'].value_counts().to_string())

    # Initialize LIME text explainer
    class_names = ['NEGATIVE', 'POSITIVE']
    explainer = LimeTextExplainer(class_names=class_names, random_state=SEED)

    def predict_proba(texts):
        """Prediction function for LIME text"""
        probs = []
        for text in texts:
            result = sentiment_pipeline(text[:512])[0]
            if result['label'] == 'POSITIVE':
                probs.append([1 - result['score'], result['score']])
            else:
                probs.append([result['score'], 1 - result['score']])
        return np.array(probs)

    # Find examples
    print("\n" + "-" * 50)
    print("Selecting reviews for LIME analysis...")
    print("-" * 50)

    # Get high confidence examples
    pos_reviews = subset_df[subset_df['sentiment'] == 'POSITIVE'].sort_values('confidence', ascending=False)
    neg_reviews = subset_df[subset_df['sentiment'] == 'NEGATIVE'].sort_values('confidence', ascending=False)

    # Select 2 positive and 2 negative reviews
    selected_reviews = pd.concat([
        pos_reviews.head(2),
        neg_reviews.head(2)
    ])

    print("\n" + "-" * 50)
    print("LIME Explanations for Selected Reviews")
    print("-" * 50)

    lime_results = []
    for i, (idx, row) in enumerate(selected_reviews.iterrows()):
        print(f"\n--- Review {i+1} ---")
        print(f"Sentiment: {row['sentiment']} (confidence: {row['confidence']:.3f})")
        print(f"Text preview: {row['text'][:200]}...")

        # Get LIME explanation
        exp = explainer.explain_instance(
            row['text'][:512],
            predict_proba,
            num_features=10,
            num_samples=100
        )
        gc.collect()

        print("\nTop words influencing prediction:")
        for word, weight in exp.as_list():
            direction = "supports" if weight > 0 else "opposes"
            print(f"  '{word}': {weight:+.4f} ({direction} {row['sentiment']})")

        lime_results.append((row, exp))

    # Visualize LIME for one example
    if lime_results:
        row, exp = lime_results[0]
        fig = exp.as_pyplot_figure()
        plt.title(f"LIME: Words affecting {row['sentiment']} prediction")
        plt.tight_layout()
        plt.savefig('ex4_lime_text_example.png', dpi=150, bbox_inches='tight')
        plt.show()

    # Flip test
    print("\n" + "-" * 50)
    print("FLIP TEST: Modifying reviews to change predictions")
    print("-" * 50)

    # Take a positive review and try to flip it
    pos_example = pos_reviews.iloc[0]
    print(f"\nOriginal POSITIVE review (conf: {pos_example['confidence']:.3f}):")
    print(f"  {pos_example['text'][:300]}...")

    # Simple modifications
    modified_text = pos_example['text']

    # Replace positive words with negative
    replacements = [
        ('great', 'terrible'), ('good', 'bad'), ('love', 'hate'),
        ('amazing', 'awful'), ('excellent', 'poor'), ('wonderful', 'horrible'),
        ('best', 'worst'), ('perfect', 'flawed'), ('recommend', 'avoid'),
        ('fantastic', 'disappointing'), ('happy', 'unhappy')
    ]

    for old, new in replacements:
        modified_text = modified_text.replace(old, new)
        modified_text = modified_text.replace(old.capitalize(), new.capitalize())

    # Get new prediction
    new_result = sentiment_pipeline(modified_text[:512])[0]
    print(f"\nModified review prediction: {new_result['label']} (conf: {new_result['score']:.3f})")
    print(f"  Modified text preview: {modified_text[:300]}...")

    if new_result['label'] != pos_example['sentiment']:
        print("\n  SUCCESS: Prediction flipped!")
    else:
        print("\n  Prediction did NOT flip (model robust to these changes)")

    # Try flipping a negative review
    neg_example = neg_reviews.iloc[0]
    print(f"\nOriginal NEGATIVE review (conf: {neg_example['confidence']:.3f}):")
    print(f"  {neg_example['text'][:300]}...")

    modified_neg = neg_example['text']

    # Replace negative words with positive
    neg_replacements = [
        ('terrible', 'great'), ('bad', 'good'), ('hate', 'love'),
        ('awful', 'amazing'), ('poor', 'excellent'), ('horrible', 'wonderful'),
        ('worst', 'best'), ('disappointing', 'fantastic'), ('avoid', 'recommend'),
        ('boring', 'exciting'), ('waste', 'value')
    ]

    for old, new in neg_replacements:
        modified_neg = modified_neg.replace(old, new)
        modified_neg = modified_neg.replace(old.capitalize(), new.capitalize())

    new_neg_result = sentiment_pipeline(modified_neg[:512])[0]
    print(f"\nModified review prediction: {new_neg_result['label']} (conf: {new_neg_result['score']:.3f})")
    print(f"  Modified text preview: {modified_neg[:300]}...")

    if new_neg_result['label'] != neg_example['sentiment']:
        print("\n  SUCCESS: Prediction flipped!")
    else:
        print("\n  Prediction did NOT flip (model robust to these changes)")

    return subset_df


# ============================================================================
# EXERCISE 5: Uncertainty and Robustness
# ============================================================================

def exercise_5_uncertainty_robustness(subset_df=None):
    """
    Analyze model confidence and robustness to noise.
    """
    print("\n" + "=" * 70)
    print("EXERCISE 5: Uncertainty and Robustness")
    print("=" * 70)

    import os
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'

    from transformers import pipeline
    import gc

    # Load data if not passed
    if subset_df is None:
        reviews_path = os.path.join(os.path.dirname(__file__), "reviews.csv")
        df = pd.read_csv(reviews_path)
        subset_df = df.sample(n=2000, random_state=SEED).reset_index(drop=True)

        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )

        texts = subset_df['text'].tolist()
        results = []
        for i, text in enumerate(texts):
            results.append(sentiment_pipeline(text[:512])[0])
            if (i + 1) % 500 == 0:
                gc.collect()

        subset_df['sentiment'] = [r['label'] for r in results]
        subset_df['confidence'] = [r['score'] for r in results]
        subset_df['text_length'] = subset_df['text'].str.len()
        subset_df['word_count'] = subset_df['text'].str.split().str.len()
    else:
        import torch

        if torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

        sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            device=device
        )

    # Analysis 1: Confidence vs Review Length
    print("\n" + "-" * 50)
    print("Analysis 1: Confidence vs Review Length")
    print("-" * 50)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    colors = ['red' if s == 'NEGATIVE' else 'blue' for s in subset_df['sentiment']]
    axes[0].scatter(subset_df['word_count'], subset_df['confidence'],
                   c=colors, alpha=0.3, s=10)
    axes[0].set_xlabel('Word Count')
    axes[0].set_ylabel('Confidence')
    axes[0].set_title('Confidence vs Word Count')
    axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% threshold')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', alpha=0.5, label='POSITIVE'),
                      Patch(facecolor='red', alpha=0.5, label='NEGATIVE')]
    axes[0].legend(handles=legend_elements)

    # Binned analysis
    bins = pd.cut(subset_df['word_count'], bins=[0, 50, 100, 200, 500, 10000])
    confidence_by_length = subset_df.groupby(bins)['confidence'].agg(['mean', 'std', 'count'])

    bin_labels = ['0-50', '50-100', '100-200', '200-500', '500+']
    axes[1].bar(bin_labels, confidence_by_length['mean'],
               yerr=confidence_by_length['std'], capsize=5, color='steelblue', alpha=0.7)
    axes[1].set_xlabel('Word Count Range')
    axes[1].set_ylabel('Mean Confidence')
    axes[1].set_title('Average Confidence by Review Length')
    axes[1].set_ylim(0.5, 1.0)

    plt.tight_layout()
    plt.savefig('ex5_confidence_vs_length.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Statistics
    print("\nConfidence statistics by review length:")
    print(confidence_by_length.to_string())

    # Correlation
    corr = subset_df['word_count'].corr(subset_df['confidence'])
    print(f"\nCorrelation between word count and confidence: {corr:.4f}")

    # Analysis 2: Robustness to Noise
    print("\n" + "-" * 50)
    print("Analysis 2: Robustness to Noise")
    print("-" * 50)

    # Sample some reviews for robustness testing
    sample_reviews = subset_df.sample(n=100, random_state=SEED)

    def add_noise(text, noise_type):
        """Add different types of noise to text"""
        words = text.split()

        if noise_type == 'polarity_swap':
            # Swap sentiment words
            swaps = {
                'good': 'bad', 'bad': 'good',
                'great': 'terrible', 'terrible': 'great',
                'love': 'hate', 'hate': 'love',
                'amazing': 'awful', 'awful': 'amazing',
                'best': 'worst', 'worst': 'best',
                'excellent': 'poor', 'poor': 'excellent'
            }
            words = [swaps.get(w.lower(), w) for w in words]

        elif noise_type == 'remove_negations':
            # Remove negation words
            negations = {'not', "n't", 'never', 'no', 'none', 'nothing'}
            words = [w for w in words if w.lower() not in negations and not w.lower().endswith("n't")]

        elif noise_type == 'typos':
            # Add random typos
            import random
            random.seed(SEED)
            new_words = []
            for w in words:
                if len(w) > 3 and random.random() < 0.1:  # 10% chance of typo
                    i = random.randint(1, len(w)-2)
                    w = w[:i] + w[i+1] + w[i] + w[i+2:]  # Swap adjacent chars
                new_words.append(w)
            words = new_words

        return ' '.join(words)

    # Test each noise type
    noise_types = ['polarity_swap', 'remove_negations', 'typos']

    results_summary = []

    for noise_type in noise_types:
        print(f"\nTesting noise type: {noise_type}")

        original_texts = sample_reviews['text'].tolist()
        noisy_texts = [add_noise(t, noise_type) for t in original_texts]

        # Get predictions for noisy texts one at a time
        noisy_results = []
        for text in noisy_texts:
            noisy_results.append(sentiment_pipeline(text[:512])[0])
        gc.collect()

        # Compare predictions
        original_preds = sample_reviews['sentiment'].tolist()
        noisy_preds = [r['label'] for r in noisy_results]

        original_conf = sample_reviews['confidence'].tolist()
        noisy_conf = [r['score'] for r in noisy_results]

        # Calculate metrics
        flips = sum(1 for o, n in zip(original_preds, noisy_preds) if o != n)
        flip_rate = flips / len(original_preds)

        avg_conf_change = np.mean([abs(o - n) for o, n in zip(original_conf, noisy_conf)])

        print(f"  Prediction flips: {flips}/{len(original_preds)} ({flip_rate:.1%})")
        print(f"  Average confidence change: {avg_conf_change:.4f}")

        results_summary.append({
            'noise_type': noise_type,
            'flip_rate': flip_rate,
            'avg_conf_change': avg_conf_change
        })

    # Plot robustness summary
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    noise_labels = [r['noise_type'] for r in results_summary]
    flip_rates = [r['flip_rate'] for r in results_summary]
    conf_changes = [r['avg_conf_change'] for r in results_summary]

    axes[0].bar(noise_labels, flip_rates, color=['coral', 'steelblue', 'green'], alpha=0.7)
    axes[0].set_ylabel('Flip Rate')
    axes[0].set_title('Prediction Flip Rate by Noise Type')
    axes[0].set_ylim(0, 1)

    axes[1].bar(noise_labels, conf_changes, color=['coral', 'steelblue', 'green'], alpha=0.7)
    axes[1].set_ylabel('Average Confidence Change')
    axes[1].set_title('Confidence Change by Noise Type')

    plt.tight_layout()
    plt.savefig('ex5_robustness.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Analysis 3: Overconfidence analysis
    print("\n" + "-" * 50)
    print("Analysis 3: Overconfidence Analysis")
    print("-" * 50)

    # Check if ground truth matches predictions
    # Assuming polarity: 1=negative, 2=positive
    subset_df['ground_truth'] = subset_df['polarity'].map({1: 'NEGATIVE', 2: 'POSITIVE'})
    subset_df['correct'] = subset_df['sentiment'] == subset_df['ground_truth']

    accuracy = subset_df['correct'].mean()
    print(f"\nOverall accuracy (model vs ground truth): {accuracy:.1%}")

    # Calibration: are high-confidence predictions more often correct?
    confidence_bins = pd.cut(subset_df['confidence'], bins=[0.5, 0.7, 0.9, 0.95, 1.0])
    calibration = subset_df.groupby(confidence_bins)['correct'].agg(['mean', 'count'])
    calibration.columns = ['accuracy', 'count']

    print("\nCalibration (accuracy by confidence level):")
    print(calibration.to_string())

    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Confidence-Length Correlation: {corr:.4f}")
    print(f"  Model Accuracy: {accuracy:.1%}")
    print(f"  Polarity swap flip rate: {results_summary[0]['flip_rate']:.1%}")
    print(f"  Negation removal flip rate: {results_summary[1]['flip_rate']:.1%}")
    print(f"  Typos flip rate: {results_summary[2]['flip_rate']:.1%}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Run all exercises."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + "EXERCISE LIST 2: EXPLAINABILITY AND UNCERTAINTY".center(68) + "#")
    print("#" + f"Student ID: ai24m033 | Seed: {SEED}".center(68) + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70 + "\n")

    # Exercise 1: Permutation Importance
    exercise_1_permutation_importance()

    # Exercise 2: Integrated Gradients
    exercise_2_integrated_gradients()

    # Exercise 3: LIME for Images
    exercise_3_lime_images()

    # Exercise 4: LIME for Text
    subset_df = exercise_4_lime_text()

    # Exercise 5: Uncertainty and Robustness
    exercise_5_uncertainty_robustness(subset_df)

    print("\n" + "=" * 70)
    print("ALL EXERCISES COMPLETED")
    print("=" * 70)
    print("\nGenerated files:")
    print("  - ex1_permutation_importance.png")
    print("  - ex1_partial_dependence.png")
    print("  - ex2_ig_*.png (integrated gradients visualizations)")
    print("  - ex3_lime_images.png")
    print("  - ex4_lime_text_example.png")
    print("  - ex5_confidence_vs_length.png")
    print("  - ex5_robustness.png")


if __name__ == "__main__":
    main()
