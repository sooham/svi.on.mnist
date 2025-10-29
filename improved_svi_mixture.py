#!/usr/bin/env python3
"""
Improved Stochastic Variational Inference for Mixture of Gaussians

This implementation addresses the component collapse issue by:
1. Better initialization strategies
2. Asymmetric and informative priors
3. Regularization terms to encourage separation
4. Adaptive learning rates
5. Temperature annealing for exploration
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import digamma, logsumexp, polygamma
from scipy.stats import dirichlet, norm
import seaborn as sns
from scipy.special import gammaln
import matplotlib.animation as animation


class ImprovedSVIMixture:
    def __init__(self, K=3, sigma_squared=20, random_seed=42):
        """
        Initialize the improved SVI mixture model
        
        Args:
            K: Number of mixture components
            sigma_squared: Known observation variance
            random_seed: Random seed for reproducibility
        """
        self.K = K
        self.sigma_squared = sigma_squared
        np.random.seed(random_seed)
        
    def initialize_parameters(self, samples, init_method='kmeans_plus'):
        """
        Initialize variational parameters with better separation strategies
        """
        n_samples = len(samples)
        data_min, data_max = samples.min(), samples.max()
        data_range = data_max - data_min
        data_std = samples.std()
        
        # More informative priors to prevent collapse
        # Use asymmetric Dirichlet prior that encourages some components to be larger
        base_priors = [2.0, 1.5, 1.0, 0.5]  # Decreasing prior weights
        self.alpha_prior = np.array(base_priors[:self.K])
        
        # Prior for component means - more diffuse to allow exploration
        self.m0 = samples.mean()
        self.s0_squared = data_std**2 * 3  # More diffuse prior
        
        print(f"Improved Prior hyperparameters:")
        print(f"  Asymmetric Dirichlet α: {self.alpha_prior}")
        print(f"  Normal m0: {self.m0:.2f}, s0²: {self.s0_squared:.2f}")
        print(f"  Known σ²: {self.sigma_squared}")
        
        # Initialize mixture weights with asymmetric prior + small noise
        self.alpha_var = self.alpha_prior + np.random.uniform(0.1, 0.5, self.K)
        
        # Initialize component means with better separation
        if init_method == 'kmeans_plus':
            self.means_var = self._kmeans_plus_init(samples)
        elif init_method == 'quantiles':
            quantiles = np.linspace(0.15, 0.85, self.K)
            self.means_var = np.quantile(samples, quantiles)
        elif init_method == 'spread':
            self.means_var = np.linspace(data_min + 0.2*data_range, 
                                       data_max - 0.2*data_range, self.K)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
        
        # Add controlled noise to avoid exact overlaps
        noise_scale = data_std / 10  # Smaller noise
        self.means_var += np.random.normal(0, noise_scale, self.K)
        
        # Initialize component variances
        self.s_squared_var = np.ones(self.K) * self.s0_squared / 3
        
        # Initialize responsibilities with slight bias toward separation
        self.phi = self._initialize_responsibilities(samples)
        
        print(f"Initial component means: {self.means_var}")
        print(f"Mean separations: {np.diff(np.sort(self.means_var))}")
        print(f"Initial weights: {self.alpha_var / self.alpha_var.sum()}")
        
    def _kmeans_plus_init(self, samples):
        """K-means++ style initialization for better separation"""
        means = np.zeros(self.K)
        data_min, data_max = samples.min(), samples.max()
        
        # First center: random point in middle 60% of data range
        range_start = data_min + 0.2 * (data_max - data_min)
        range_end = data_max - 0.2 * (data_max - data_min)
        means[0] = np.random.uniform(range_start, range_end)
        
        # Subsequent centers: maximize minimum distance to existing centers
        for k in range(1, self.K):
            candidates = np.random.uniform(data_min, data_max, 200)
            min_distances = np.zeros(200)
            
            for i, candidate in enumerate(candidates):
                distances = np.abs(candidate - means[:k])
                min_distances[i] = np.min(distances)
            
            # Choose candidate with maximum minimum distance
            best_idx = np.argmax(min_distances)
            means[k] = candidates[best_idx]
        
        return means
    
    def _initialize_responsibilities(self, samples):
        """Initialize responsibilities with slight bias toward component separation"""
        n_samples = len(samples)
        phi = np.zeros((n_samples, self.K))
        
        # Assign each sample to closest component with some noise
        for i, x in enumerate(samples):
            distances = np.abs(x - self.means_var)
            # Soft assignment based on inverse distance
            weights = 1.0 / (distances + 1e-6)
            weights = weights / weights.sum()
            
            # Add some randomness to avoid hard assignments
            noise = np.random.dirichlet(np.ones(self.K) * 0.1)
            phi[i] = 0.8 * weights + 0.2 * noise
            phi[i] = phi[i] / phi[i].sum()  # Renormalize
        
        return phi
    
    def compute_elbo_with_regularization(self, samples, temperature=1.0, separation_penalty=0.1):
        """
        Compute ELBO with regularization terms to prevent collapse
        
        Args:
            samples: Data samples
            temperature: Temperature for annealing (higher = more exploration)
            separation_penalty: Penalty coefficient for component overlap
        """
        n_samples = len(samples)
        elbo = 0.0
        
        # Standard ELBO terms
        elbo += self._compute_likelihood_term(samples, temperature)
        elbo += self._compute_prior_terms()
        elbo += self._compute_entropy_terms()
        
        # Regularization: Penalty for component overlap
        if separation_penalty > 0:
            separation_term = self._compute_separation_penalty(separation_penalty)
            elbo += separation_term
        
        return elbo
    
    def _compute_likelihood_term(self, samples, temperature=1.0):
        """Compute E_q[log p(x | z, μ)] with temperature scaling"""
        n_samples = len(samples)
        likelihood_term = 0.0
        
        for i in range(n_samples):
            for k in range(self.K):
                log_likelihood = norm.logpdf(samples[i], 
                                           loc=self.means_var[k], 
                                           scale=np.sqrt(self.sigma_squared))
                likelihood_term += self.phi[i, k] * log_likelihood / temperature
        
        return likelihood_term
    
    def _compute_prior_terms(self):
        """Compute prior terms E_q[log p(π)] + E_q[log p(μ)]"""
        prior_term = 0.0
        
        # E_q[log p(π)] - Dirichlet prior
        digamma_sum = digamma(self.alpha_var.sum())
        prior_term += gammaln(self.alpha_prior.sum()) - gammaln(self.alpha_prior).sum()
        
        for k in range(self.K):
            expected_log_pi = digamma(self.alpha_var[k]) - digamma_sum
            prior_term += (self.alpha_prior[k] - 1) * expected_log_pi
        
        # E_q[log p(μ)] - Normal prior for component means
        for k in range(self.K):
            expected_squared_diff = (self.means_var[k] - self.m0)**2 + self.s_squared_var[k]
            log_prior = -0.5 * np.log(2 * np.pi * self.s0_squared) - 0.5 * expected_squared_diff / self.s0_squared
            prior_term += log_prior
        
        return prior_term
    
    def _compute_entropy_terms(self):
        """Compute entropy terms -E_q[log q(z)] - E_q[log q(π)] - E_q[log q(μ)]"""
        entropy = 0.0
        
        # Entropy of responsibilities: -E_q[log q(z)]
        for i in range(len(self.phi)):
            for k in range(self.K):
                if self.phi[i, k] > 1e-10:
                    entropy -= self.phi[i, k] * np.log(self.phi[i, k])
        
        # Entropy of Dirichlet: -E_q[log q(π)]
        digamma_sum = digamma(self.alpha_var.sum())
        entropy -= gammaln(self.alpha_var.sum()) - gammaln(self.alpha_var).sum()
        for k in range(self.K):
            expected_log_pi = digamma(self.alpha_var[k]) - digamma_sum
            entropy -= (self.alpha_var[k] - 1) * expected_log_pi
        
        # Entropy of Gaussians: -E_q[log q(μ)]
        for k in range(self.K):
            entropy += 0.5 * np.log(2 * np.pi * np.e * self.s_squared_var[k])
        
        return entropy
    
    def _compute_separation_penalty(self, penalty_coeff):
        """
        Compute penalty term that encourages component separation
        
        Penalty is higher when components are closer together
        """
        penalty = 0.0
        
        for i in range(self.K):
            for j in range(i + 1, self.K):
                # Distance between component means
                distance = abs(self.means_var[i] - self.means_var[j])
                # Penalty decreases exponentially with distance
                penalty -= penalty_coeff * np.exp(-distance / 10.0)
        
        return penalty
    
    def compute_gradients(self, samples_batch, temperature=1.0, separation_penalty=0.1):
        """
        Compute gradients of regularized ELBO w.r.t. all variational parameters
        """
        batch_size = len(samples_batch)
        
        # Gradients w.r.t. Dirichlet parameters (mixture weights)
        grad_alpha = self._compute_alpha_gradients(samples_batch, temperature)
        
        # Gradients w.r.t. component means
        grad_means = self._compute_means_gradients(samples_batch, temperature, separation_penalty)
        
        # Gradients w.r.t. component variances
        grad_variances = self._compute_variance_gradients()
        
        # Gradients w.r.t. responsibilities (local parameters)
        grad_phi = self._compute_phi_gradients(samples_batch, temperature)
        
        return grad_alpha, grad_means, grad_variances, grad_phi
    
    def _compute_alpha_gradients(self, samples_batch, temperature):
        """Compute gradients w.r.t. Dirichlet parameters"""
        batch_size = len(samples_batch)
        grad_alpha = np.zeros(self.K)
        
        alpha_sum = self.alpha_var.sum()
        digamma_sum = digamma(alpha_sum)
        trigamma_sum = polygamma(1, alpha_sum)
        
        for k in range(self.K):
            digamma_k = digamma(self.alpha_var[k])
            trigamma_k = polygamma(1, self.alpha_var[k])
            
            # From likelihood and prior terms
            expected_count = self.phi[:batch_size, k].sum()  # Only use batch
            grad_alpha[k] += expected_count * (digamma_k - digamma_sum) / temperature
            grad_alpha[k] += (self.alpha_prior[k] - 1) * (digamma_k - digamma_sum)
            
            # From entropy terms
            grad_alpha[k] -= (digamma_k - digamma_sum)
            grad_alpha[k] -= (self.alpha_var[k] - 1) * (trigamma_k - trigamma_sum)
        
        return np.clip(grad_alpha, -10.0, 10.0)  # Gradient clipping
    
    def _compute_means_gradients(self, samples_batch, temperature, separation_penalty):
        """Compute gradients w.r.t. component means"""
        batch_size = len(samples_batch)
        grad_means = np.zeros(self.K)
        
        for k in range(self.K):
            # Likelihood term
            likelihood_grad = 0.0
            for i in range(batch_size):
                likelihood_grad += self.phi[i, k] * (samples_batch[i] - self.means_var[k])
            likelihood_grad = likelihood_grad / (self.sigma_squared * temperature)
            
            # Prior term
            prior_grad = -(self.means_var[k] - self.m0) / self.s0_squared
            
            # Separation penalty gradient
            separation_grad = 0.0
            if separation_penalty > 0:
                for j in range(self.K):
                    if j != k:
                        distance = self.means_var[k] - self.means_var[j]
                        sign = np.sign(distance)
                        exp_term = np.exp(-abs(distance) / 10.0)
                        separation_grad += separation_penalty * sign * exp_term / 10.0
            
            grad_means[k] = likelihood_grad + prior_grad + separation_grad
        
        return np.clip(grad_means, -50.0, 50.0)  # Gradient clipping
    
    def _compute_variance_gradients(self):
        """Compute gradients w.r.t. component variances"""
        grad_variances = np.zeros(self.K)
        
        for k in range(self.K):
            # Prior term
            prior_grad = -1.0 / (2 * self.s0_squared)
            # Entropy term
            entropy_grad = 1.0 / (2 * self.s_squared_var[k])
            grad_variances[k] = prior_grad + entropy_grad
        
        return np.clip(grad_variances, -5.0, 5.0)
    
    def _compute_phi_gradients(self, samples_batch, temperature):
        """Compute gradients w.r.t. responsibilities"""
        batch_size = len(samples_batch)
        grad_phi = np.zeros((batch_size, self.K))
        
        alpha_sum = self.alpha_var.sum()
        digamma_sum = digamma(alpha_sum)
        
        for i in range(batch_size):
            log_phi_unnorm = np.zeros(self.K)
            
            for k in range(self.K):
                # Likelihood term
                log_likelihood = norm.logpdf(samples_batch[i], 
                                           loc=self.means_var[k], 
                                           scale=np.sqrt(self.sigma_squared))
                
                # Prior term (expected log π_k)
                expected_log_pi = digamma(self.alpha_var[k]) - digamma_sum
                
                log_phi_unnorm[k] = (log_likelihood + expected_log_pi) / temperature
            
            # Numerically stable softmax gradient
            max_log = np.max(log_phi_unnorm)
            log_phi_unnorm -= max_log
            
            exp_vals = np.exp(log_phi_unnorm)
            softmax_vals = exp_vals / np.sum(exp_vals)
            
            grad_phi[i] = softmax_vals - self.phi[i]
        
        return grad_phi
    
    def fit(self, samples, n_iterations=300, batch_size=500, 
            learning_rate=0.01, phi_lr=0.05, 
            temperature_schedule='linear', separation_penalty=0.5,
            verbose=True):
        """
        Fit the mixture model using improved SVI
        
        Args:
            samples: Data samples
            n_iterations: Number of optimization iterations
            batch_size: Mini-batch size
            learning_rate: Learning rate for global parameters
            phi_lr: Learning rate for local parameters (responsibilities)
            temperature_schedule: 'linear', 'exponential', or 'constant'
            separation_penalty: Coefficient for separation regularization
            verbose: Whether to print progress
        """
        n_samples = len(samples)
        
        # Storage for tracking convergence
        self.elbo_history = []
        self.mean_history = []
        self.weight_history = []
        self.gradient_norms = []
        
        if verbose:
            print(f"\nRunning Improved Stochastic Variational Inference...")
            print(f"Iterations: {n_iterations}, Batch size: {batch_size}")
            print(f"Global LR: {learning_rate}, Local LR: {phi_lr}")
            print(f"Separation penalty: {separation_penalty}")
        
        for iteration in range(n_iterations):
            # Temperature annealing for exploration
            if temperature_schedule == 'linear':
                temperature = max(0.1, 1.0 - 0.9 * iteration / n_iterations)
            elif temperature_schedule == 'exponential':
                temperature = max(0.1, np.exp(-3 * iteration / n_iterations))
            else:
                temperature = 1.0
            
            # Adaptive learning rates
            current_lr = learning_rate / (1 + 0.001 * iteration)
            current_phi_lr = phi_lr / (1 + 0.0005 * iteration)
            
            # Sample mini-batch
            batch_indices = np.random.choice(n_samples, min(batch_size, n_samples), replace=False)
            samples_batch = samples[batch_indices]
            
            # Compute gradients
            grad_alpha, grad_means, grad_variances, grad_phi = self.compute_gradients(
                samples_batch, temperature, separation_penalty
            )
            
            # Scale gradients for unbiased estimates
            scale = n_samples / len(samples_batch)
            grad_alpha *= scale
            grad_means *= scale
            
            # Gradient ascent updates (maximize ELBO)
            self.alpha_var += current_lr * grad_alpha
            self.means_var += current_lr * grad_means
            self.s_squared_var += current_lr * grad_variances
            
            # Update local parameters (responsibilities)
            phi_batch_new = self.phi[batch_indices] + current_phi_lr * grad_phi
            
            # Apply softmax to maintain probability constraints
            for i in range(len(batch_indices)):
                # Numerically stable softmax
                max_val = np.max(phi_batch_new[i])
                phi_batch_new[i] = phi_batch_new[i] - max_val
                phi_batch_new[i] = np.exp(phi_batch_new[i])
                phi_sum = np.sum(phi_batch_new[i])
                if phi_sum > 1e-10:
                    phi_batch_new[i] /= phi_sum
                else:
                    phi_batch_new[i] = np.ones(self.K) / self.K
            
            self.phi[batch_indices] = phi_batch_new
            
            # Ensure parameter constraints
            self.alpha_var = np.clip(self.alpha_var, 0.1, 100.0)
            self.s_squared_var = np.clip(self.s_squared_var, 1e-6, 1000.0)
            self.means_var = np.clip(self.means_var, samples.min() - 50, samples.max() + 50)
            
            # Track convergence
            grad_norm = np.sqrt(np.sum(grad_alpha**2) + np.sum(grad_means**2) + np.sum(grad_variances**2))
            self.gradient_norms.append(grad_norm)
            
            # Store parameters for tracking at every iteration
            elbo = self.compute_elbo_with_regularization(samples, temperature, separation_penalty)
            self.elbo_history.append(elbo)
            
            estimated_weights = self.alpha_var / self.alpha_var.sum()
            self.mean_history.append(self.means_var.copy())
            self.weight_history.append(estimated_weights.copy())
            
            if iteration % 20 == 0 or iteration == n_iterations - 1:
                
                if verbose:
                    print(f"Iter {iteration:3d}: ELBO = {elbo:10.2f}, T = {temperature:.3f}, ||∇|| = {grad_norm:.4f}")
                    print(f"  Means: {self.means_var}")
                    print(f"  Weights: {estimated_weights}")
                    print(f"  Separations: {np.diff(np.sort(self.means_var))}")
                
                # Early stopping if gradients become too small
                if grad_norm < 1e-6 and iteration > 50:
                    if verbose:
                        print(f"Converged at iteration {iteration} (gradient norm < 1e-6)")
                    break
        
        return {
            'alpha': self.alpha_var,
            'means': self.means_var,
            'variances': self.s_squared_var,
            'weights': self.alpha_var / self.alpha_var.sum(),
            'elbo_history': self.elbo_history,
            'mean_history': self.mean_history,
            'weight_history': self.weight_history,
            'gradient_norms': self.gradient_norms,
            'phi': self.phi
        }
    
    def create_convergence_animation(self, samples, true_means, true_weights, true_stds, 
                                   save_path='mixture_convergence.gif', fps=3, dpi=150):
        """
        Create an animation showing the convergence of mixture components compared to true distribution
        
        Args:
            samples: Original data samples
            true_means: True component means
            true_weights: True component weights  
            true_stds: True component standard deviations
            save_path: Path to save the animation
            fps: Frames per second
            dpi: Resolution
        """
        if not hasattr(self, 'mean_history') or len(self.mean_history) == 0:
            print("No convergence history available. Run fit() first.")
            return None
            
        print(f"Creating convergence animation with {len(self.mean_history)} frames...")
        
        # Set up the figure with larger size for better visibility
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        # Data range for plotting - extend beyond data range to show full distributions
        data_min, data_max = samples.min(), samples.max()
        data_range = data_max - data_min
        x_min = data_min - 0.3 * data_range
        x_max = data_max + 0.3 * data_range
        x_range = np.linspace(x_min, x_max, 1000)
        
        # Colors for components
        colors_true = ['red', 'orange', 'brown', 'pink', 'purple']
        colors_est = ['blue', 'cyan', 'navy', 'lightblue', 'darkblue']
        
        # Pre-compute true mixture for efficiency
        true_mixture = sum(w * norm.pdf(x_range, m, s) 
                          for w, m, s in zip(true_weights, true_means, true_stds))
        
        # Find maximum density to set consistent y-limits (don't clip peaks)
        max_density = np.max(true_mixture) * 1.2  # Add 20% padding
        
        # Also check estimated mixtures to ensure we don't clip anything
        for frame_idx in range(len(self.mean_history)):
            current_means = self.mean_history[frame_idx]
            current_weights = self.weight_history[frame_idx]
            
            est_mixture = sum(w * norm.pdf(x_range, m, np.sqrt(self.sigma_squared)) 
                             for w, m in zip(current_weights, current_means))
            frame_max = np.max(est_mixture)
            if frame_max > max_density / 1.2:
                max_density = frame_max * 1.2
        
        print(f"Animation y-limit set to {max_density:.4f} to avoid clipping")
        
        def animate(frame):
            # Clear both axes
            ax1.clear()
            ax2.clear()
            
            # Calculate current iteration (every iteration is saved)
            iteration = frame
            
            # Top plot: Data histogram and evolving mixture
            ax1.hist(samples, bins=60, density=True, alpha=0.3, color='gray', 
                    label='Data', range=(data_min, data_max))
            
            # Plot true mixture (static)
            ax1.plot(x_range, true_mixture, 'r-', linewidth=4, label='True mixture', alpha=0.8)
            
            # Plot individual true components
            for i, (w, m, s) in enumerate(zip(true_weights, true_means, true_stds)):
                if i < len(colors_true):
                    component_pdf = w * norm.pdf(x_range, m, s)
                    ax1.plot(x_range, component_pdf, '--', color=colors_true[i], 
                            linewidth=2, alpha=0.7, label=f'True μ={m}, π={w:.2f}')
            
            # Plot current estimated mixture
            if frame < len(self.mean_history):
                current_means = self.mean_history[frame]
                current_weights = self.weight_history[frame]
                
                # Plot estimated mixture
                est_mixture = sum(w * norm.pdf(x_range, m, np.sqrt(self.sigma_squared)) 
                                 for w, m in zip(current_weights, current_means))
                ax1.plot(x_range, est_mixture, 'b-', linewidth=4, 
                        label='Estimated mixture', alpha=0.9)
                
                # Plot individual estimated components
                for i, (w, m) in enumerate(zip(current_weights, current_means)):
                    if i < len(colors_est):
                        component_pdf = w * norm.pdf(x_range, m, np.sqrt(self.sigma_squared))
                        ax1.plot(x_range, component_pdf, ':', color=colors_est[i], 
                                linewidth=3, alpha=0.8, 
                                label=f'Est μ={m:.1f}, π={w:.2f}')
                
                # Add vertical lines for component means
                for i, m in enumerate(current_means):
                    if i < len(colors_est):
                        ax1.axvline(x=m, color=colors_est[i], linestyle='-', 
                                   alpha=0.4, linewidth=2)
                
                # Add vertical lines for true means
                for i, m in enumerate(true_means):
                    if i < len(colors_true):
                        ax1.axvline(x=m, color=colors_true[i], linestyle='--', 
                                   alpha=0.6, linewidth=2)
            
            # Set limits and labels for top plot
            ax1.set_xlim(x_min, x_max)
            ax1.set_ylim(0, max_density)
            ax1.set_xlabel('Value', fontsize=12)
            ax1.set_ylabel('Density', fontsize=12)
            ax1.set_title(f'Mixture Model Convergence - Iteration {iteration}', fontsize=14, fontweight='bold')
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax1.grid(True, alpha=0.3)
            
            # Bottom plot: Mean evolution over time
            iterations_so_far = np.arange(0, frame + 1)
            
            # Plot true means as horizontal lines
            for i, true_mean in enumerate(true_means):
                if i < len(colors_true):
                    ax2.axhline(y=true_mean, color=colors_true[i], linestyle='--', 
                               alpha=0.8, linewidth=3, label=f'True μ{i+1}={true_mean}')
            
            # Plot estimated mean evolution
            if frame < len(self.mean_history):
                for k in range(len(current_means)):
                    if k < len(colors_est):
                        means_so_far = [self.mean_history[i][k] for i in range(frame + 1)]
                        ax2.plot(iterations_so_far, means_so_far, 'o-', 
                                color=colors_est[k], linewidth=3, markersize=5, 
                                label=f'Est μ{k+1}', alpha=0.8)
                        
                        # Highlight current position with larger marker
                        if len(means_so_far) > 0:
                            ax2.plot(iterations_so_far[-1], means_so_far[-1], 'o', 
                                    color=colors_est[k], markersize=10, 
                                    markeredgecolor='black', markeredgewidth=2)
                            
                            # Add text annotation for current value
                            ax2.annotate(f'{means_so_far[-1]:.1f}', 
                                        (iterations_so_far[-1], means_so_far[-1]),
                                        xytext=(8, 8), textcoords='offset points',
                                        fontsize=10, fontweight='bold',
                                        bbox=dict(boxstyle='round,pad=0.3', 
                                                facecolor=colors_est[k], alpha=0.3))
            
            # Set limits and labels for bottom plot
            ax2.set_xlim(0, len(self.mean_history))
            
            # Dynamic y-limits based on data
            all_means = true_means + [m for means in self.mean_history[:frame+1] for m in means]
            if all_means:
                y_margin = (max(all_means) - min(all_means)) * 0.1
                ax2.set_ylim(min(all_means) - y_margin, max(all_means) + y_margin)
            
            ax2.set_xlabel('Iteration', fontsize=12)
            ax2.set_ylabel('Component Mean', fontsize=12)
            ax2.set_title('Evolution of Component Means', fontsize=14, fontweight='bold')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Add ELBO and separation info as text
            if frame < len(self.elbo_history):
                current_elbo = self.elbo_history[frame]
                separations = np.diff(np.sort(current_means))
                min_sep = separations.min() if len(separations) > 0 else 0
                
                info_text = f'ELBO: {current_elbo:.1f}\nMin separation: {min_sep:.2f}'
                ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, 
                        fontsize=11, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                        verticalalignment='top', fontweight='bold')
            
            plt.tight_layout()
        
        # Create animation
        n_frames = len(self.mean_history)
        print(f"Animating {n_frames} frames...")
        
        anim = animation.FuncAnimation(fig, animate, frames=n_frames, 
                                     interval=1000//fps, blit=False, repeat=True)
        
        # Save animation
        print(f"Saving animation to {save_path}...")
        try:
            anim.save(save_path, writer='pillow', fps=fps, dpi=dpi)
            print(f"Animation saved successfully!")
        except Exception as e:
            print(f"Error saving animation: {e}")
            print("Trying alternative writer...")
            try:
                anim.save(save_path.replace('.gif', '.mp4'), writer='ffmpeg', fps=fps, dpi=dpi)
                print(f"Animation saved as MP4!")
            except:
                print("Could not save animation. Displaying instead...")
        
        plt.show()
        return anim


def test_improved_svi():
    """Test the improved SVI implementation"""
    # Generate test data (same as in notebook)
    np.random.seed(38)
    means = [30, 50, 50, 100]
    stds = [np.sqrt(20), np.sqrt(20), np.sqrt(100), np.sqrt(20)]
    weights = [0.5, 0.35, 0.03, 0.12]
    
    # Sample from the mixture
    n_samples = 50000
    component_samples = np.random.choice(len(means), size=n_samples, p=weights)
    samples = np.array([np.random.normal(means[i], stds[i]) for i in component_samples])
    
    print("="*60)
    print("TESTING IMPROVED SVI MIXTURE MODEL")
    print("="*60)
    print(f"True means: {means}")
    print(f"True weights: {weights}")
    print(f"True std devs: {[f'{std:.2f}' for std in stds]}")
    
    # Initialize and fit improved model
    model = ImprovedSVIMixture(K=4, sigma_squared=20, random_seed=42)  # Use K=4 to match true model
    model.initialize_parameters(samples, init_method='kmeans_plus')
    
    results = model.fit(samples, n_iterations=400, batch_size=1000, 
                       learning_rate=0.005, phi_lr=0.02,
                       temperature_schedule='linear', separation_penalty=1.0,
                       verbose=True)
    
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(f"Estimated means: {results['means']}")
    print(f"Estimated weights: {results['weights']}")
    print(f"Mean separations: {np.diff(np.sort(results['means']))}")
    
    # Create convergence animation
    print("\n" + "="*60)
    print("CREATING CONVERGENCE ANIMATION")
    print("="*60)
    
    animation_obj = model.create_convergence_animation(
        samples, means, weights, stds,
        save_path='improved_mixture_convergence.gif',
        fps=4, dpi=150
    )
    
    return results, samples, means, weights, stds


if __name__ == "__main__":
    results, samples, true_means, true_weights, true_stds = test_improved_svi()
