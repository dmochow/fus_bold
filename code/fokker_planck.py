import numpy as np
import pandas as pd
from scipy import optimize
from scipy.stats import gaussian_kde
import warnings
from typing import Tuple, Dict, Optional, List

class FokkerPlanckFitter:
    """
    Fits 1-D Fokker-Planck equation to time series data.
    
    The 1-D Fokker-Planck equation describes the evolution of probability density:
    ∂P/∂t = -∂/∂x[D1(x)P] + (1/2)∂²/∂x²[D2(x)P]
    
    Where:
    - D1(x) is the drift coefficient 
    - D2(x) is the diffusion coefficient
    
    This implementation assumes polynomial forms:
    - D1(x) = a1*x + a0 (linear drift)
    - D2(x) = b1*x + b0 (linear diffusion, with constraint b0 > 0)
    """
    
    def __init__(self, drift_order: int = 1, diffusion_order: int = 0):
        """
        Initialize the Fokker-Planck fitter.
        
        Parameters:
        -----------
        drift_order : int
            Polynomial order for drift coefficient (default: 1 for linear)
        diffusion_order : int  
            Polynomial order for diffusion coefficient (default: 0 for constant)
        """
        self.drift_order = drift_order
        self.diffusion_order = diffusion_order
        self.fitted_params = None
        self.fit_result = None
        
    def _estimate_coefficients_kramers_moyal(self, x: np.ndarray, dt: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate drift and diffusion coefficients using Kramers-Moyal expansion.
        
        Parameters:
        -----------
        x : np.ndarray
            Time series data
        dt : float
            Time step
            
        Returns:
        --------
        drift_coeff : np.ndarray
            Estimated drift coefficients
        diff_coeff : np.ndarray  
            Estimated diffusion coefficients
        """
        # Calculate increments
        dx = np.diff(x)
        x_mid = x[:-1]  # Use left endpoints for binning
        
        # Create bins for conditional averages
        n_bins = min(20, len(x) // 10)  # Adaptive binning
        bins = np.linspace(x_mid.min(), x_mid.max(), n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Estimate moments conditionally
        drift_est = np.zeros(len(bin_centers))
        diff_est = np.zeros(len(bin_centers))
        
        for i, (left, right) in enumerate(zip(bins[:-1], bins[1:])):
            mask = (x_mid >= left) & (x_mid < right)
            if np.sum(mask) > 1:  # Need at least 2 points
                drift_est[i] = np.mean(dx[mask]) / dt
                diff_est[i] = np.var(dx[mask]) / dt
            else:
                # Fallback for sparse bins
                drift_est[i] = 0
                diff_est[i] = np.var(dx) / dt
                
        return bin_centers, drift_est, diff_est
    
    def _fit_polynomial_coefficients(self, x_vals: np.ndarray, y_vals: np.ndarray, 
                                   order: int) -> np.ndarray:
        """Fit polynomial coefficients to data."""
        if len(x_vals) < order + 1:
            # Not enough points for polynomial fit
            if order == 0:
                return np.array([np.mean(y_vals)])
            else:
                # Return zeros for higher order terms
                coeffs = np.zeros(order + 1)
                coeffs[0] = np.mean(y_vals)
                return coeffs
        
        return np.polyfit(x_vals, y_vals, order)
    
    def fit(self, time_series: np.ndarray, dt: float = 1.0) -> Dict:
        """
        Fit Fokker-Planck parameters to time series.
        
        Parameters:
        -----------
        time_series : np.ndarray
            1-D time series data
        dt : float
            Time step (default: 1.0)
            
        Returns:
        --------
        results : Dict
            Dictionary containing fitted parameters and diagnostics
        """
        # Validate input
        if len(time_series) < 10:
            raise ValueError("Time series too short for reliable fitting")
        
        # Standardize time series for numerical stability
        x_mean = np.mean(time_series)
        x_std = np.std(time_series)
        if x_std == 0:
            raise ValueError("Time series has zero variance")
            
        x_norm = (time_series - x_mean) / x_std
        
        # Estimate coefficients using Kramers-Moyal
        x_vals, drift_est, diff_est = self._estimate_coefficients_kramers_moyal(x_norm, dt)
        
        # Fit polynomial forms
        drift_coeffs = self._fit_polynomial_coefficients(x_vals, drift_est, self.drift_order)
        diff_coeffs = self._fit_polynomial_coefficients(x_vals, diff_est, self.diffusion_order)
        
        # Ensure diffusion coefficient is positive
        if self.diffusion_order == 0:
            diff_coeffs[0] = max(diff_coeffs[0], 1e-6)
        
        # Transform coefficients back to original scale
        # For drift: D1_orig(x) = D1_norm((x-mean)/std) / std
        # For diffusion: D2_orig(x) = D2_norm((x-mean)/std) / std^2
        drift_coeffs_orig = drift_coeffs.copy()
        diff_coeffs_orig = diff_coeffs.copy()
        
        if self.drift_order >= 1:
            drift_coeffs_orig[-1] /= x_std  # Linear term
        if self.drift_order >= 0:
            drift_coeffs_orig[0] = drift_coeffs[0] / x_std - drift_coeffs[-1] * x_mean / x_std
            
        diff_coeffs_orig /= (x_std ** 2)
        
        # Store results
        self.fitted_params = {
            'drift_coeffs': drift_coeffs_orig,
            'diffusion_coeffs': diff_coeffs_orig,
            'x_mean': x_mean,
            'x_std': x_std
        }
        
        # Calculate diagnostics
        diagnostics = self._calculate_diagnostics(time_series, dt)
        
        results = {
            'drift_coeffs': drift_coeffs_orig,
            'diffusion_coeffs': diff_coeffs_orig,
            'diagnostics': diagnostics,
            'x_mean': x_mean,
            'x_std': x_std
        }
        
        self.fit_result = results
        return results
    
    def _calculate_diagnostics(self, time_series: np.ndarray, dt: float) -> Dict:
        """Calculate diagnostic measures for fit quality."""
        diagnostics = {}
        
        # Basic statistics
        diagnostics['n_points'] = len(time_series)
        diagnostics['mean'] = np.mean(time_series)
        diagnostics['std'] = np.std(time_series)
        diagnostics['skewness'] = self._calculate_skewness(time_series)
        diagnostics['kurtosis'] = self._calculate_kurtosis(time_series)
        
        # Stationarity test (simple)
        mid_point = len(time_series) // 2
        first_half_mean = np.mean(time_series[:mid_point])
        second_half_mean = np.mean(time_series[mid_point:])
        diagnostics['stationarity_test'] = abs(first_half_mean - second_half_mean) / np.std(time_series)
        
        return diagnostics
    
    def _calculate_skewness(self, x: np.ndarray) -> float:
        """Calculate skewness."""
        return np.mean(((x - np.mean(x)) / np.std(x)) ** 3)
    
    def _calculate_kurtosis(self, x: np.ndarray) -> float:
        """Calculate excess kurtosis."""
        return np.mean(((x - np.mean(x)) / np.std(x)) ** 4) - 3
    
    def get_drift_function(self, x_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get drift function values over specified range.
        
        Parameters:
        -----------
        x_range : np.ndarray, optional
            Range of x values. If None, uses reasonable range around data.
            
        Returns:
        --------
        x_vals : np.ndarray
            X values
        drift_vals : np.ndarray
            Drift function values
        """
        if self.fitted_params is None:
            raise ValueError("Must fit model first")
            
        if x_range is None:
            x_mean = self.fitted_params['x_mean']
            x_std = self.fitted_params['x_std']
            x_range = np.linspace(x_mean - 3*x_std, x_mean + 3*x_std, 100)
        
        drift_coeffs = self.fitted_params['drift_coeffs']
        drift_vals = np.polyval(drift_coeffs[::-1], x_range)  # polyval expects descending order
        
        return x_range, drift_vals
    
    def get_diffusion_function(self, x_range: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get diffusion function values over specified range.
        
        Parameters:
        -----------
        x_range : np.ndarray, optional
            Range of x values. If None, uses reasonable range around data.
            
        Returns:
        --------
        x_vals : np.ndarray
            X values  
        diffusion_vals : np.ndarray
            Diffusion function values
        """
        if self.fitted_params is None:
            raise ValueError("Must fit model first")
            
        if x_range is None:
            x_mean = self.fitted_params['x_mean']
            x_std = self.fitted_params['x_std']
            x_range = np.linspace(x_mean - 3*x_std, x_mean + 3*x_std, 100)
        
        diff_coeffs = self.fitted_params['diffusion_coeffs']
        diff_vals = np.polyval(diff_coeffs[::-1], x_range)
        
        # Ensure positive diffusion
        diff_vals = np.maximum(diff_vals, 1e-6)
        
        return x_range, diff_vals


def fit_time_series_by_periods(time_series: np.ndarray, 
                              period_boundaries: List[int] = [300, 600],
                              dt: float = 1.0,
                              drift_order: int = 1,
                              diffusion_order: int = 0) -> Dict:
    """
    Fit Fokker-Planck model to different periods of a time series.
    
    Parameters:
    -----------
    time_series : np.ndarray
        Full time series (900 timepoints for FUS experiment)
    period_boundaries : List[int]
        Boundaries between periods (default: [300, 600] for baseline/stim/recovery)
    dt : float
        Time step
    drift_order : int
        Polynomial order for drift
    diffusion_order : int
        Polynomial order for diffusion
        
    Returns:
    --------
    results : Dict
        Results for each period ('baseline', 'stimulation', 'recovery')
    """
    # Define periods
    periods = {
        'baseline': (0, period_boundaries[0]),
        'stimulation': (period_boundaries[0], period_boundaries[1]), 
        'recovery': (period_boundaries[1], len(time_series))
    }
    
    results = {}
    
    for period_name, (start, end) in periods.items():
        period_data = time_series[start:end]
        
        try:
            fitter = FokkerPlanckFitter(drift_order=drift_order, 
                                      diffusion_order=diffusion_order)
            fit_result = fitter.fit(period_data, dt=dt)
            fit_result['fitter'] = fitter
            fit_result['period_bounds'] = (start, end)
            results[period_name] = fit_result
            
        except Exception as e:
            warnings.warn(f"Failed to fit {period_name} period: {str(e)}")
            results[period_name] = {'error': str(e)}
    
    return results