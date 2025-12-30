import os
from typing import Any, Dict, List

import mlflow
import mlflow.pyfunc
import pandas as pd
from mlflow.exceptions import RestException
from mlflow.tracking import MlflowClient
from sklearn.base import BaseEstimator, TransformerMixin

# CRITICAL: Patch _check_unknown IMMEDIATELY after sklearn import to catch it before OneHotEncoder imports it
try:
    import sklearn.utils._encode as encode_module
    if hasattr(encode_module, '_check_unknown'):
        original_check_unknown = encode_module._check_unknown
        
        def patched_check_unknown(X, known_values, *, return_mask=False, xp=None):
            import numpy as np
            # Convert all inputs to strings to avoid type comparison issues
            # CRITICAL: Convert both X and known_values to ensure type consistency
            try:
                # Convert X to strings if it's object dtype (might have mixed types)
                if isinstance(X, np.ndarray):
                    if X.dtype == object:
                        X = np.array([str(x) if x is not None and pd.notna(x) and str(x) not in ['nan', 'None', '<NA>', ''] else 'Unknown' 
                                     for x in X.flatten()], dtype=object).reshape(X.shape)
                    # Also convert numeric arrays if known_values is object (mixed scenario)
                    elif X.dtype.kind in ['f', 'i'] and isinstance(known_values, np.ndarray) and known_values.dtype == object:
                        X = np.array([str(x) if pd.notna(x) else 'Unknown' for x in X.flatten()], dtype=object).reshape(X.shape)
                
                # ALWAYS convert known_values to strings if X is object/string
                # This ensures type consistency
                if isinstance(known_values, np.ndarray):
                    if known_values.dtype == object:
                        known_values = np.array([str(x) if x is not None and pd.notna(x) else 'Unknown' 
                                                for x in known_values.flatten()], dtype=object)
                    elif isinstance(X, np.ndarray) and X.dtype == object:
                        # If X is object/string, convert known_values to match
                        known_values = np.array([str(x) if pd.notna(x) else 'Unknown' 
                                                for x in known_values.flatten()], dtype=object)
            except Exception as e:
                # If conversion fails, try to ensure both are same type
                try:
                    if isinstance(X, np.ndarray) and isinstance(known_values, np.ndarray):
                        if X.dtype == object or known_values.dtype == object:
                            # If either is object, convert both to strings
                            X = np.array([str(x) for x in X.flatten()], dtype=object).reshape(X.shape)
                            known_values = np.array([str(x) for x in known_values.flatten()], dtype=object)
                except:
                    pass
            
            # Call original but catch isnan error for string arrays
            # Handle both old and new sklearn versions (xp parameter might not exist)
            try:
                import inspect
                sig = inspect.signature(original_check_unknown)
                if 'xp' in sig.parameters:
                    return original_check_unknown(X, known_values, return_mask=return_mask, xp=xp)
                else:
                    # Older sklearn version without xp parameter
                    return original_check_unknown(X, known_values, return_mask=return_mask)
            except (TypeError, ValueError) as e:
                if 'isnan' in str(e) or 'not supported' in str(e):
                    # Handle string arrays - skip isnan check and use set operations
                    import numpy as np
                    if isinstance(X, np.ndarray):
                        unique_X = np.unique(X)
                    else:
                        unique_X = np.unique(np.array(X))
                    if isinstance(known_values, np.ndarray):
                        known_set = set(known_values.flatten())
                    else:
                        known_set = set(np.array(known_values).flatten())
                    diff = np.array([x for x in unique_X if x not in known_set], dtype=object)
                    if return_mask:
                        if isinstance(X, np.ndarray):
                            valid_mask = np.array([x in known_set for x in X.flatten()], dtype=bool).reshape(X.shape)
                        else:
                            valid_mask = np.array([x in known_set for x in X], dtype=bool)
                        return diff, valid_mask
                    return diff
                raise
        
        encode_module._check_unknown = patched_check_unknown
        sklearn.utils._encode._check_unknown = patched_check_unknown
except Exception:
    pass  # Silently fail if patch can't be applied

from src.config import get_mlflow_config


def patch_model_feature_engineer(model: mlflow.pyfunc.PyFuncModel) -> mlflow.pyfunc.PyFuncModel:
    """
    Patch the FeatureEngineer in the model's pipeline to fix Categorical dtype issues.
    This is a workaround for old models that create Categorical columns which cause
    issues in OneHotEncoder.
    """
    import sys
    import numpy as np
    
    # CRITICAL: Patch _check_unknown BEFORE any sklearn modules import it
    # This must happen first, before sklearn.preprocessing._encoders imports it
    try:
        import sklearn.utils._encode as encode_module
        if hasattr(encode_module, '_check_unknown'):
            original_check_unknown_func = encode_module._check_unknown
            
            def patched_check_unknown_func(X, known_values, *, return_mask=False, xp=None):
                import numpy as np
                # Convert all inputs to strings to avoid type comparison issues
                try:
                    # Convert X
                    if isinstance(X, np.ndarray) and X.dtype == object:
                        X = np.array([str(x) if x is not None and pd.notna(x) and str(x) not in ['nan', 'None', '<NA>', ''] else 'Unknown' 
                                     for x in X.flatten()], dtype=object).reshape(X.shape)
                    # Convert known_values  
                    if isinstance(known_values, np.ndarray) and known_values.dtype == object:
                        known_values = np.array([str(x) if x is not None and pd.notna(x) else 'Unknown' 
                                                for x in known_values.flatten()], dtype=object)
                except:
                    pass
                
                # Call original but catch isnan error for string arrays
                try:
                    return original_check_unknown_func(X, known_values, return_mask=return_mask, xp=xp)
                except (TypeError, ValueError) as e:
                    if 'isnan' in str(e) or 'not supported' in str(e):
                        # Handle string arrays - skip isnan check
                        if isinstance(X, np.ndarray):
                            unique_X = np.unique(X)
                        else:
                            unique_X = np.unique(np.array(X))
                        if isinstance(known_values, np.ndarray):
                            known_set = set(known_values.flatten())
                        else:
                            known_set = set(np.array(known_values).flatten())
                        diff = np.array([x for x in unique_X if x not in known_set], dtype=object)
                        if return_mask:
                            if isinstance(X, np.ndarray):
                                valid_mask = np.array([x in known_set for x in X.flatten()], dtype=bool).reshape(X.shape)
                            else:
                                valid_mask = np.array([x in known_set for x in X], dtype=bool)
                            return diff, valid_mask
                        return diff
                    raise
            
            encode_module._check_unknown = patched_check_unknown_func
            # Also patch in the module dict to catch direct imports
            import sklearn.utils._encode
            sklearn.utils._encode._check_unknown = patched_check_unknown_func
            print("✓ Monkey-patched sklearn.utils._encode._check_unknown at module level", flush=True)
            sys.stdout.flush()
    except Exception as e:
        print(f"Warning: Could not patch _check_unknown at module level: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    # Monkey-patch numpy's argsort to handle mixed types gracefully
    try:
        import numpy as np
        from numpy.core import fromnumeric
        
        original_argsort = fromnumeric.argsort
        def patched_argsort(a, axis=-1, kind=None, order=None):
            # If array has object dtype and might have mixed types, convert to strings first
            try:
                if isinstance(a, np.ndarray) and a.dtype == object:
                    # Check if array has mixed types by trying to sort
                    try:
                        # Try normal argsort - if it fails, convert to strings
                        return original_argsort(a, axis=axis, kind=kind, order=order)
                    except TypeError:
                        # Mixed types detected - convert all to strings
                        a_str = np.array([str(x) if x is not None and pd.notna(x) else 'Unknown' 
                                         for x in a.flatten()], dtype=object).reshape(a.shape)
                        return original_argsort(a_str, axis=axis, kind=kind, order=order)
            except:
                pass
            return original_argsort(a, axis=axis, kind=kind, order=order)
        
        fromnumeric.argsort = patched_argsort
        # Also patch the wrapper
        if hasattr(np, 'argsort'):
            np.argsort = patched_argsort
        print("✓ Monkey-patched numpy.argsort to handle mixed types", flush=True)
        sys.stdout.flush()
    except Exception as e:
        print(f"Warning: Could not monkey-patch numpy.argsort: {e}", flush=True)
        sys.stdout.flush()
    
    # Monkey-patch sklearn's _check_unknown and related functions to handle mixed types
    try:
        from sklearn.utils import _encode
        from sklearn.externals.array_api_extra._lib import _funcs
        
        # Patch setdiff1d which calls argsort and causes the error
        if hasattr(_funcs, 'setdiff1d'):
            original_setdiff1d = _funcs.setdiff1d
            def patched_setdiff1d(x1, x2, *, assume_unique=False, xp=None):
                # Convert to strings before setdiff1d to avoid type comparison issues
                import numpy as np
                try:
                    if isinstance(x1, np.ndarray) and x1.dtype == object:
                        x1 = np.array([str(x) if x is not None and pd.notna(x) else 'Unknown' for x in x1.flatten()], dtype=object).reshape(x1.shape)
                    if isinstance(x2, np.ndarray) and x2.dtype == object:
                        x2 = np.array([str(x) if x is not None and pd.notna(x) else 'Unknown' for x in x2.flatten()], dtype=object).reshape(x2.shape)
                except:
                    pass
                return original_setdiff1d(x1, x2, assume_unique=assume_unique, xp=xp)
            _funcs.setdiff1d = patched_setdiff1d
            print("✓ Monkey-patched sklearn setdiff1d to handle mixed types", flush=True)
        
        # Patch _check_unknown in multiple places where it might be imported
        # Patch in sklearn.utils._encode
        if hasattr(_encode, '_check_unknown'):
            original_check_unknown = _encode._check_unknown
        else:
            # Try importing it directly
            try:
                from sklearn.utils._encode import _check_unknown as original_check_unknown
            except:
                original_check_unknown = None
        
        if original_check_unknown is not None:
            
            def patched_check_unknown(X, known_values, *, return_mask=False, xp=None):
                # AGGRESSIVE FIX: Convert ALL values to strings immediately to avoid any type comparison issues
                import numpy as np
                converted_to_str = False
                try:
                    # Convert X to strings
                    if isinstance(X, np.ndarray):
                        if X.dtype == object:
                            # Object array - convert all elements to strings
                            X = np.array([str(x) if x is not None and pd.notna(x) and str(x) not in ['nan', 'None', '<NA>', ''] else 'Unknown' 
                                         for x in X.flatten()], dtype=object).reshape(X.shape)
                            converted_to_str = True
                        elif X.dtype.kind in ['f', 'i']:  # float or int - might be mixed
                            # Check if might have mixed types by trying a sample comparison
                            try:
                                # Try to see if there are any string-like values
                                sample = X.flatten()[:10]
                                has_str = any(isinstance(x, str) for x in sample)
                                if has_str:
                                    X = np.array([str(x) if pd.notna(x) else 'Unknown' for x in X.flatten()], dtype=object).reshape(X.shape)
                                    converted_to_str = True
                            except:
                                pass
                    
                    # Convert known_values to strings
                    if isinstance(known_values, np.ndarray):
                        if known_values.dtype == object:
                            known_values = np.array([str(x) if x is not None and pd.notna(x) else 'Unknown' 
                                                    for x in known_values.flatten()], dtype=object)
                            converted_to_str = True
                        elif known_values.dtype.kind in ['f', 'i']:
                            # Check for mixed types
                            try:
                                sample = known_values.flatten()[:10]
                                has_str = any(isinstance(x, str) for x in sample)
                                if has_str:
                                    known_values = np.array([str(x) if pd.notna(x) else 'Unknown' for x in known_values.flatten()], dtype=object)
                                    converted_to_str = True
                            except:
                                pass
                    
                    # Also handle list/tuple inputs
                    if isinstance(X, (list, tuple)):
                        X = [str(x) if x is not None and pd.notna(x) else 'Unknown' for x in X]
                        converted_to_str = True
                    if isinstance(known_values, (list, tuple)):
                        known_values = [str(x) if x is not None and pd.notna(x) else 'Unknown' for x in known_values]
                        converted_to_str = True
                        
                except Exception as e:
                    print(f"Warning in patched_check_unknown conversion: {e}", flush=True)
                
                # If we converted to strings, we need to handle the isnan check differently
                # The original _check_unknown tries to check for NaN, but strings don't have NaN
                if converted_to_str:
                    # For string arrays, we can skip the NaN check and go straight to setdiff1d
                    # But we need to replicate the logic without the isnan check
                    try:
                        import numpy as np
                        # Get unique values from X
                        if isinstance(X, np.ndarray):
                            unique_X = np.unique(X)
                        else:
                            unique_X = np.unique(np.array(X))
                        
                        # Find differences (values in X but not in known_values)
                        if isinstance(known_values, np.ndarray):
                            known_set = set(known_values.flatten())
                        else:
                            known_set = set(np.array(known_values).flatten())
                        
                        diff = np.array([x for x in unique_X if x not in known_set])
                        
                        if return_mask:
                            # Create mask for valid (known) values
                            if isinstance(X, np.ndarray):
                                valid_mask = np.array([x in known_set for x in X.flatten()]).reshape(X.shape)
                            else:
                                valid_mask = np.array([x in known_set for x in X])
                            return diff, valid_mask
                        else:
                            return diff
                    except Exception as e:
                        print(f"Warning: Custom _check_unknown failed, falling back: {e}", flush=True)
                        # Fallback to original but catch the isnan error
                        try:
                            return original_check_unknown(X, known_values, return_mask=return_mask, xp=xp)
                        except TypeError:
                            # If isnan fails, treat as no unknown values
                            if return_mask:
                                return np.array([]), np.ones_like(X, dtype=bool)
                            return np.array([])
                
                # Call original function with string-converted data
                return original_check_unknown(X, known_values, return_mask=return_mask, xp=xp)
            
            _encode._check_unknown = patched_check_unknown
            print("✓ Monkey-patched sklearn.utils._encode._check_unknown to handle mixed types", flush=True)
            sys.stdout.flush()
        else:
            print("Warning: _check_unknown not found in sklearn.utils._encode", flush=True)
    except Exception as e:
        print(f"Warning: Could not monkey-patch _check_unknown: {e}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    print("DEBUG: patch_model_feature_engineer called", flush=True)
    try:
        print(f"DEBUG: model type: {type(model)}", flush=True)
        print(f"DEBUG: model attributes: {[attr for attr in dir(model) if not attr.startswith('_')][:10]}", flush=True)
        
        # MLflow models have different structures - try multiple ways to access the sklearn model
        sklearn_model = None
        if hasattr(model, 'sklearn_model'):
            sklearn_model = model.sklearn_model
            print(f"DEBUG: Found sklearn_model via .sklearn_model", flush=True)
        elif hasattr(model, '_model_impl'):
            sklearn_model = model._model_impl.sklearn_model if hasattr(model._model_impl, 'sklearn_model') else None
            print(f"DEBUG: Found sklearn_model via ._model_impl", flush=True)
        elif hasattr(model, 'model'):
            sklearn_model = model.model
            print(f"DEBUG: Found sklearn_model via .model", flush=True)
        
        if sklearn_model is None:
            print("DEBUG: Could not find sklearn_model, trying to access directly", flush=True)
            # Try to get the underlying model
            if hasattr(model, 'predict'):
                # The model itself might be the sklearn model wrapped
                sklearn_model = model
                print(f"DEBUG: Using model itself as sklearn_model", flush=True)
        
        if sklearn_model is None:
            print("WARNING: Could not find sklearn_model to patch", flush=True)
            return model
            
        print(f"DEBUG: sklearn_model type: {type(sklearn_model)}", flush=True)
        
        # If it's a Pipeline, find and patch both FeatureEngineer and ColumnTransformer
        if hasattr(sklearn_model, 'steps'):
                print(f"DEBUG: Found pipeline with {len(sklearn_model.steps)} steps", flush=True)
                for i, (name, step) in enumerate(sklearn_model.steps):
                    class_name = step.__class__.__name__ if hasattr(step, '__class__') else ''
                    print(f"DEBUG: Step {i}: name='{name}', class='{class_name}'", flush=True)
                    
                    # Check if this is the FeatureEngineer step
                    if 'FeatureEngineer' in class_name:
                        # Store the original transform method
                        original_transform = step.transform
                        
                        # Create a patched version that wraps the original
                        def patched_transform(X):
                            result = original_transform(X)
                            # Convert ALL object and categorical dtype columns to strings
                            # This ensures no mixed types reach OneHotEncoder
                            if isinstance(result, pd.DataFrame):
                                fixed_cols = []
                                for col in result.columns:
                                    dtype_str = str(result[col].dtype)
                                    original_dtype = result[col].dtype
                                    # Convert Categorical to string
                                    if hasattr(result[col].dtype, 'categories') or dtype_str == 'category':
                                        result[col] = result[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
                                        fixed_cols.append(f"{col}({original_dtype}->str)")
                                    # Convert ALL object dtype columns to strings (to handle any mixed types)
                                    elif dtype_str == 'object':
                                        # Check for mixed types first
                                        try:
                                            sample_vals = result[col].dropna().head(10)
                                            has_numeric = any(isinstance(x, (int, float)) and not isinstance(x, bool) for x in sample_vals if pd.notna(x))
                                            has_string = any(isinstance(x, str) for x in sample_vals if pd.notna(x))
                                            if has_numeric and has_string:
                                                fixed_cols.append(f"{col}(mixed->str)")
                                        except:
                                            pass
                                        # Force all object columns to strings - this handles mixed types
                                        result[col] = result[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown').replace('None', 'Unknown')
                                if fixed_cols:
                                    print(f"DEBUG: FeatureEngineer fixed columns: {', '.join(fixed_cols)}", flush=True)
                            return result
                        
                        # Replace the transform method
                        step.transform = patched_transform
                        print(f"✓ Patched FeatureEngineer step '{name}' to fix categorical dtype issues", flush=True)
                        sys.stdout.flush()
                    
                    # Also patch ColumnTransformer to ensure categorical columns are strings
                    elif 'ColumnTransformer' in class_name:
                        original_ct_transform = step.transform
                        def patched_ct_transform(X):
                            # BEFORE ColumnTransformer processes, ensure all object/categorical columns are strings
                            if isinstance(X, pd.DataFrame):
                                X_fixed = X.copy()
                                for col in X_fixed.columns:
                                    dtype_str = str(X_fixed[col].dtype)
                                    # Convert Categorical to string
                                    if hasattr(X_fixed[col].dtype, 'categories') or dtype_str == 'category':
                                        X_fixed[col] = X_fixed[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
                                    # Convert object columns - ensure they're all strings (handle mixed types)
                                    elif dtype_str == 'object':
                                        # Force all values to strings to avoid mixed type issues
                                        X_fixed[col] = X_fixed[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown').replace('None', 'Unknown')
                                # Call original transform with fixed data
                                return original_ct_transform(X_fixed)
                            else:
                                # If not DataFrame, try to convert to DataFrame, fix, then convert back
                                X_df = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
                                X_fixed = X_df.copy()
                                for col in X_fixed.columns:
                                    dtype_str = str(X_fixed[col].dtype)
                                    if hasattr(X_fixed[col].dtype, 'categories') or dtype_str == 'category' or dtype_str == 'object':
                                        X_fixed[col] = X_fixed[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
                                return original_ct_transform(X_fixed)
                        step.transform = patched_ct_transform
                        print(f"✓ Patched ColumnTransformer step '{name}' to fix categorical dtype issues", flush=True)
                        sys.stdout.flush()
                        
                        # Also patch the transformers inside ColumnTransformer
                        if hasattr(step, 'transformers'):
                            for trans_name, trans, cols in step.transformers:
                                # Patch OneHotEncoder if it exists
                                if hasattr(trans, '__class__') and 'OneHotEncoder' in trans.__class__.__name__:
                                    # CRITICAL: Check categories_ dtype and ensure input matches
                                    # Only convert if categories are already object/string, don't force conversion
                                    if hasattr(trans, 'categories_') and trans.categories_ is not None:
                                        import numpy as np
                                        try:
                                            print(f"DEBUG: OneHotEncoder categories_ type: {type(trans.categories_)}", flush=True)
                                            if isinstance(trans.categories_, (tuple, list)) and len(trans.categories_) > 0:
                                                first_cat = trans.categories_[0]
                                                print(f"DEBUG: First category array dtype: {first_cat.dtype if hasattr(first_cat, 'dtype') else type(first_cat)}", flush=True)
                                                
                                                # Only convert if categories are already object dtype (string-like)
                                                # If they're numeric, we should NOT convert - keep input numeric
                                                needs_string_conversion = False
                                                if isinstance(first_cat, np.ndarray) and first_cat.dtype == object:
                                                    needs_string_conversion = True
                                                    # Clean up existing string categories
                                                    converted_cats = []
                                                    for cat_array in trans.categories_:
                                                        if isinstance(cat_array, np.ndarray) and cat_array.dtype == object:
                                                            # Clean strings: strip whitespace, replace empty with 'Unknown'
                                                            cat_str = np.array([str(x).strip() if x is not None and pd.notna(x) and str(x).strip() else 'Unknown' 
                                                                               for x in cat_array.flatten()], dtype=object)
                                                            converted_cats.append(cat_str)
                                                        else:
                                                            converted_cats.append(cat_array)
                                                    trans.categories_ = tuple(converted_cats)
                                                    print(f"✓ Cleaned OneHotEncoder string categories_", flush=True)
                                                else:
                                                    print(f"DEBUG: Categories are numeric ({first_cat.dtype if hasattr(first_cat, 'dtype') else 'unknown'}), keeping as-is", flush=True)
                                        except Exception as e:
                                            print(f"Warning: Could not check/convert categories_: {e}", flush=True)
                                            import traceback
                                            traceback.print_exc()
                                    
                                    # Patch the transform method
                                    original_ohe_transform = trans.transform
                                    def patched_ohe_transform(X):
                                        # Convert to DataFrame if needed, fix types, convert back
                                        import numpy as np
                                        if isinstance(X, np.ndarray):
                                            # Convert array to DataFrame, fix, convert back
                                            X_df = pd.DataFrame(X)
                                            for col in X_df.columns:
                                                X_df[col] = X_df[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
                                            X_fixed = X_df.values
                                        elif isinstance(X, pd.DataFrame):
                                            X_fixed = X.copy()
                                            for col in X_fixed.columns:
                                                X_fixed[col] = X_fixed[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
                                        else:
                                            X_fixed = X
                                        return original_ohe_transform(X_fixed)
                                    trans.transform = patched_ohe_transform
                                    
                                    # Also patch the internal _transform method where the error actually occurs
                                    # CRITICAL: Ensure Xi is created with object dtype when categories_ are strings
                                    if hasattr(trans, '_transform'):
                                        original_ohe_internal_transform = trans._transform
                                        
                                        # Check if ANY categories_ contain strings (not just the first)
                                        import numpy as np
                                        has_string_categories = False
                                        if hasattr(trans, 'categories_') and trans.categories_ is not None:
                                            try:
                                                # Check ALL category arrays, not just the first
                                                for cat_array in trans.categories_:
                                                    if isinstance(cat_array, np.ndarray) and cat_array.dtype == object:
                                                        has_string_categories = True
                                                        break
                                                # Also check if any category values are actually strings (even if dtype is numeric)
                                                if not has_string_categories:
                                                    for cat_array in trans.categories_:
                                                        if isinstance(cat_array, np.ndarray) and len(cat_array) > 0:
                                                            # Check first few values
                                                            sample = cat_array.flatten()[:5]
                                                            if any(isinstance(x, str) for x in sample):
                                                                has_string_categories = True
                                                                break
                                            except Exception as e:
                                                print(f"DEBUG: Error checking categories: {e}", flush=True)
                                        # More detailed debugging
                                        if hasattr(trans, 'categories_') and trans.categories_ is not None:
                                            try:
                                                print(f"DEBUG: categories_ type: {type(trans.categories_)}, length: {len(trans.categories_) if hasattr(trans.categories_, '__len__') else 'N/A'}", flush=True)
                                                for idx, cat_array in enumerate(trans.categories_[:3]):  # Check first 3
                                                    print(f"DEBUG: categories_[{idx}] type: {type(cat_array)}, dtype: {getattr(cat_array, 'dtype', 'N/A')}", flush=True)
                                                    if hasattr(cat_array, '__getitem__') and len(cat_array) > 0:
                                                        first_val = cat_array[0]
                                                        print(f"DEBUG: categories_[{idx}][0] = {repr(first_val)}, type: {type(first_val)}", flush=True)
                                            except Exception as e:
                                                print(f"DEBUG: Error in detailed check: {e}", flush=True)
                                        print(f"DEBUG: OneHotEncoder has_string_categories={has_string_categories}", flush=True)
                                        
                                        def patched_ohe_internal_transform(X, handle_unknown='error'):
                                            import numpy as np
                                            
                                            # ALWAYS check categories at runtime, not just at patch time
                                            # Categories might be mixed (some numeric, some string)
                                            runtime_has_strings = False
                                            if hasattr(trans, 'categories_') and trans.categories_ is not None:
                                                for cat_array in trans.categories_:
                                                    if hasattr(cat_array, '__getitem__') and len(cat_array) > 0:
                                                        first_val = cat_array[0]
                                                        if isinstance(first_val, str):
                                                            runtime_has_strings = True
                                                            break
                                            
                                            # Convert input based on runtime check
                                            if isinstance(X, np.ndarray):
                                                if runtime_has_strings or X.dtype == object:
                                                    # Convert to strings - safer for mixed categories
                                                    X = np.array([str(x).strip() if x is not None and pd.notna(x) and str(x).strip() else 'Unknown' 
                                                                 for x in X.flatten()], dtype=object).reshape(X.shape)
                                                    # Also ensure dtype_ is object to match
                                                    if hasattr(trans, 'dtype_'):
                                                        trans.dtype_ = object
                                            
                                            # Try original transform with string input
                                            try:
                                                return original_ohe_internal_transform(X, handle_unknown=handle_unknown)
                                            except ValueError as ve:
                                                if 'could not convert string to float' in str(ve):
                                                    # Force object dtype and retry
                                                    original_dtype = getattr(trans, 'dtype_', None)
                                                    trans.dtype_ = object
                                                    try:
                                                        result = original_ohe_internal_transform(X, handle_unknown=handle_unknown)
                                                        if original_dtype is not None:
                                                            trans.dtype_ = original_dtype
                                                        return result
                                                    except:
                                                        if original_dtype is not None:
                                                            trans.dtype_ = original_dtype
                                                        raise
                                                raise
                                        
                                        # Use MethodType to ensure proper binding
                                        import types
                                        bound_method = types.MethodType(patched_ohe_internal_transform, trans)
                                        trans._transform = bound_method
                                        # Also try patching the class if it exists
                                        try:
                                            from sklearn.preprocessing._encoders import OneHotEncoder as OHEClass
                                            if not hasattr(OHEClass._transform, '_patched'):
                                                original_class_transform = OHEClass._transform
                                                def patched_class_transform(self, X, handle_unknown='error', **kwargs):
                                                    import numpy as np
                                                    # Check if this instance has string categories and clean them
                                                    has_str = False
                                                    if hasattr(self, 'categories_') and self.categories_ is not None:
                                                        cleaned_cats = []
                                                        for cat_array in self.categories_:
                                                            if hasattr(cat_array, '__getitem__') and len(cat_array) > 0:
                                                                first_val = cat_array[0]
                                                                if isinstance(first_val, str):
                                                                    has_str = True
                                                                    # Clean category array: strip spaces, replace empty with 'Unknown'
                                                                    if isinstance(cat_array, np.ndarray):
                                                                        cleaned = np.array([str(x).strip() if x is not None and str(x).strip() else 'Unknown' 
                                                                                           for x in cat_array.flatten()], dtype=object)
                                                                        cleaned_cats.append(cleaned)
                                                                    else:
                                                                        cleaned_cats.append([str(x).strip() if x and str(x).strip() else 'Unknown' for x in cat_array])
                                                                else:
                                                                    cleaned_cats.append(cat_array)
                                                        if has_str and cleaned_cats:
                                                            self.categories_ = tuple(cleaned_cats)
                                                            # Force object dtype
                                                            if hasattr(self, 'dtype_'):
                                                                self.dtype_ = object
                                                    
                                                    # CRITICAL: Set dtype_ BEFORE calling original_transform
                                                    # This ensures Xi is created with object dtype
                                                    if has_str:
                                                        original_dtype = getattr(self, 'dtype_', None)
                                                        self.dtype_ = object
                                                    
                                                    # Convert input to object dtype if we have string categories
                                                    if has_str and isinstance(X, np.ndarray):
                                                        if X.dtype != object:
                                                            X = np.array([str(x).strip() if pd.notna(x) and str(x).strip() else 'Unknown' 
                                                                         for x in X.flatten()], dtype=object).reshape(X.shape)
                                                        else:
                                                            # Clean string input
                                                            X = np.array([str(x).strip() if x is not None and pd.notna(x) and str(x).strip() else 'Unknown' 
                                                                         for x in X.flatten()], dtype=object).reshape(X.shape)
                                                    
                                                    try:
                                                        return original_class_transform(self, X, handle_unknown=handle_unknown, **kwargs)
                                                    finally:
                                                        # Restore original dtype if we changed it
                                                        if has_str and original_dtype is not None:
                                                            self.dtype_ = original_dtype
                                                OHEClass._transform = patched_class_transform
                                                OHEClass._transform._patched = True
                                                print(f"✓ Patched OneHotEncoder class _transform method", flush=True)
                                        except Exception as e:
                                            print(f"Warning: Could not patch OneHotEncoder class: {e}", flush=True)
                                        
                                        print(f"✓ Patched OneHotEncoder instance _transform (has_string_categories={has_string_categories})", flush=True)
                                        sys.stdout.flush()
                                    
                                    print(f"✓ Patched OneHotEncoder in ColumnTransformer", flush=True)
                                    sys.stdout.flush()
                
                # Return model after patching
                if sklearn_model is not None:
                    return model
    except Exception as patch_error:
        print(f"Warning: Could not patch model pipeline: {patch_error}", flush=True)
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
    
    return model


class CategoricalFixer(BaseEstimator, TransformerMixin):
    """Wrapper transformer to fix Categorical dtype issues in old models."""
    
    def __init__(self, base_model):
        self.base_model = base_model
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Convert any Categorical dtype columns to strings
        if hasattr(X, 'columns'):
            X_fixed = X.copy()
            for col in X_fixed.columns:
                if hasattr(X_fixed[col].dtype, 'categories') or str(X_fixed[col].dtype) == 'category':
                    X_fixed[col] = X_fixed[col].astype(str).replace('nan', 'Unknown').replace('<NA>', 'Unknown')
            return X_fixed
        return X
    
    def predict(self, X):
        # Apply the fix before prediction
        X_fixed = self.transform(X) if hasattr(self, 'transform') else X
        return self.base_model.predict(X_fixed)


def load_production_model() -> mlflow.pyfunc.PyFuncModel:
    """
    Load production model from MLflow.
    
    Attempts to load from model registry first (models:/{name}/{stage}).
    Falls back to loading from the latest run if registry is not supported
    (e.g., DagsHub limitations).
    """
    cfg = get_mlflow_config()
    mlflow.set_tracking_uri(cfg.tracking_uri)

    # Optional: if running against DagsHub, you can use a token via env vars
    token = os.getenv("DAGSHUB_TOKEN")
    if token:
        os.environ["MLFLOW_TRACKING_USERNAME"] = token
        os.environ["MLFLOW_TRACKING_PASSWORD"] = token

    model_name = os.getenv("MLFLOW_MODEL_NAME", "ChurnModel")
    stage = os.getenv("MLFLOW_MODEL_STAGE", "Production")
    
    # Option to force loading from latest run instead of registry
    # Useful when model registry transitions aren't supported (e.g., DagsHub)
    use_latest_run = os.getenv("MLFLOW_USE_LATEST_RUN", "false").lower() == "true"
    
    if not use_latest_run:
        # Try to load from model registry first
        try:
            uri = f"models:/{model_name}/{stage}"
            model = mlflow.pyfunc.load_model(uri)
            print(f"Loaded model from registry: {uri}")
            
            # Try to get model version info for debugging
            try:
                client = MlflowClient()
                latest_versions = client.get_latest_versions(model_name, stages=[stage])
                if latest_versions:
                    version_info = latest_versions[0]
                    print(f"Model version: {version_info.version}, Run ID: {version_info.run_id}, Created: {version_info.creation_timestamp}")
            except Exception as e:
                print(f"Could not get model version info: {e}")
            
            # Patch the model to fix categorical dtype issues
            model = patch_model_feature_engineer(model)
            
            return model
        except RestException as e:
            # Check if this is an unsupported endpoint error (DagsHub limitation)
            error_str = str(e).lower()
            if "unsupported endpoint" in error_str or "dagshub" in error_str or "not found" in error_str:
                print(f"Warning: Model registry not supported or model not found. Falling back to latest run.")
    
    # If use_latest_run is True, or if registry failed, load from latest run
    if use_latest_run:
        print("Loading from latest run (MLFLOW_USE_LATEST_RUN=true)")
    
    # Fallback: Load from latest run in the experiment
    try:
        mlflow.set_experiment(cfg.experiment_name)
        client = MlflowClient()
        
        # Search for the latest run with the model artifact
        experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
        if experiment is None:
            raise ValueError(f"Experiment '{cfg.experiment_name}' not found.")
        
        # Get all runs, sorted by start_time descending
        runs = client.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=10
        )
        
        # Try to load model from each run until we find one
        for run in runs:
            try:
                        run_uri = f"runs:/{run.info.run_id}/model"
                        model = mlflow.pyfunc.load_model(run_uri)
                        print(f"Loaded model from run: {run.info.run_id} (fallback from registry)")
                        print(f"Run start time: {run.info.start_time}, Run name: {run.info.run_name}")
                        # Patch the model to fix categorical dtype issues
                        model = patch_model_feature_engineer(model)
                        return model
            except Exception as e:
                # Try next run if this one doesn't have a model
                print(f"Failed to load from run {run.info.run_id}: {e}")
                continue
        
        raise ValueError(
            f"Could not find a model artifact in any recent runs. "
            f"Please ensure at least one training run has logged a model."
        )
    except Exception as e2:
        if use_latest_run:
            raise ValueError(f"Failed to load model from latest run: {e2}")
        else:
            raise ValueError(
                f"Failed to load model from registry and fallback to latest run also failed: {e2}"
            )
    except Exception as e:
        # For any other exception, try the fallback
        print(f"Warning: Model loading from registry failed ({type(e).__name__}). Trying fallback.")
        try:
            mlflow.set_experiment(cfg.experiment_name)
            client = MlflowClient()
            
            experiment = mlflow.get_experiment_by_name(cfg.experiment_name)
            if experiment is None:
                raise ValueError(f"Experiment '{cfg.experiment_name}' not found.")
            
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=10
            )
            
            for run in runs:
                try:
                    run_uri = f"runs:/{run.info.run_id}/model"
                    model = mlflow.pyfunc.load_model(run_uri)
                    print(f"Loaded model from run: {run.info.run_id} (fallback)")
                    return model
                except Exception:
                    continue
            
            raise ValueError(
                f"Could not find a model artifact in any recent runs. "
                f"Please ensure at least one training run has logged a model."
            )
        except Exception as e2:
            raise ValueError(
                f"Failed to load model from registry and fallback also failed: {e2}"
            )


def predict_from_json(model: mlflow.pyfunc.PyFuncModel, payload: Dict[str, Any]) -> List[float]:
    """
    Accepts either:
    - {"records": [ {feature: value, ...}, ... ]}
    - or a single record dict {feature: value, ...}
    """
    import sys
    print("=" * 50, flush=True)
    print("predict_from_json called", flush=True)
    print(f"Payload keys: {list(payload.keys()) if isinstance(payload, dict) else 'not a dict'}", flush=True)
    sys.stdout.flush()
    
    if "records" in payload:
        df = pd.DataFrame(payload["records"])
    else:
        df = pd.DataFrame([payload])
    
    print(f"DataFrame created with shape: {df.shape}", flush=True)
    print(f"Initial columns: {list(df.columns)}", flush=True)
    sys.stdout.flush()

    # Handle ID columns: The model expects customerID if it was in training data.
    # The ColumnTransformer was fitted with customerID, so we need to provide it.
    id_column_variants = [
        "customerID",
        "customer_id",
        "CustomerID",
        "Customer_ID",
        "customerId",
        "CustomerId",
    ]

    # Check if any ID column variant exists
    existing_id_col = None
    for col in id_column_variants:
        if col in df.columns:
            existing_id_col = col
            break

    # If customerID is missing but model expects it, add/rename it.
    # The OneHotEncoder will handle unseen values (handle_unknown="ignore").
    if "customerID" not in df.columns:
        if existing_id_col:
            # Rename existing ID column to customerID
            df = df.rename(columns={existing_id_col: "customerID"})
        else:
            # Add dummy customerID - unique per row
            df["customerID"] = "pred_" + pd.Series(range(len(df))).astype(str)

    # CRITICAL: Convert types IMMEDIATELY to match training data types
    # The ColumnTransformer selects columns by dtype, so types must match exactly
    # IMPORTANT: Only handle INPUT columns - FeatureEngineer creates derived columns inside the pipeline
    
    # Define INPUT categorical columns (raw data from user - will be OneHotEncoded)
    # These must be object dtype (string) to match training
    input_categorical_cols = [
        "customerID",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "Partner",
        "Dependents",
        "gender",
    ]
    
    # Convert INPUT categorical columns to strings (object dtype)
    # This must happen BEFORE FeatureEngineer runs
    for col in input_categorical_cols:
        if col in df.columns:
            # Force to string, handling None/NaN and ensuring all values are strings
            df[col] = df[col].astype(str).replace('nan', 'Unknown').replace('None', 'Unknown')
            # Ensure all values are actually strings (handle any mixed types)
            df[col] = df[col].apply(lambda x: str(x) if pd.notna(x) and str(x) not in ['nan', 'None', ''] else 'Unknown')
        else:
            # If column is missing, add it with a default value
            df[col] = 'Unknown'
    
    # Ensure categorical columns are object dtype
    for col in input_categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('object')

    # Define INPUT numeric columns (raw data from user - will be StandardScaled)
    # These must be int64/float64 to match training
    input_numeric_cols = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "SeniorCitizen",
    ]
    
    # Convert INPUT numeric columns to proper numeric types
    for col in input_numeric_cols:
        if col in df.columns:
            # Convert to numeric, coercing errors to NaN, then fill NaN with 0
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype('float64')
        else:
            # If column is missing, add it with default value 0
            df[col] = 0.0

    # Debug: Log column dtypes before prediction (helpful for troubleshooting)
    import sys
    print(f"DataFrame dtypes before prediction (input columns only):", flush=True)
    sys.stdout.flush()
    for col in input_categorical_cols + input_numeric_cols:
        if col in df.columns:
            print(f"  {col}: {df[col].dtype} (sample value: {df[col].iloc[0] if len(df) > 0 else 'N/A'})", flush=True)
    sys.stdout.flush()
    
    print(f"Total columns in DataFrame: {len(df.columns)}", flush=True)
    print(f"All columns: {list(df.columns)}", flush=True)
    sys.stdout.flush()

    # Note: Model should already be patched by patch_model_feature_engineer() when loaded
    # Now run prediction with error handling
    try:
        preds = model.predict(df)
    except Exception as e:
        # Log detailed error information
        print(f"ERROR during prediction: {type(e).__name__}: {str(e)}", flush=True)
        print(f"DataFrame shape: {df.shape}", flush=True)
        print(f"DataFrame dtypes:\n{df.dtypes}", flush=True)
        print(f"DataFrame info:\n{df.info()}", flush=True)
        sys.stdout.flush()
        raise

    # Ensure list of floats (probabilities or labels depending on model)
    if hasattr(preds, "tolist"):
        return preds.tolist()
    return list(preds)

