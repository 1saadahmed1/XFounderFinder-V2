"""
Utility functions and classes for managing application state and data.
"""
import streamlit as st
import logging
from typing import Dict, Any, List, Set, Optional

logger = logging.getLogger(__name__)

class StateManager:
    """
    Manages the application's state using Streamlit's session_state.
    Provides a simple interface for getting and setting state variables.
    """
    
    def __init__(self):
        """
        Initializes the state manager.
        """
        if 'initialized' not in st.session_state:
            st.session_state['initialized'] = True
            
    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves a value from the session state.
        
        Args:
            key: The key of the state variable.
            default: The default value to return if the key is not found.
            
        Returns:
            The value of the state variable or the default value.
        """
        return st.session_state.get(key, default)
        
    def set(self, key: str, value: Any):
        """
        Sets a value in the session state.
        
        Args:
            key: The key of the state variable.
            value: The value to set.
        """
        st.session_state[key] = value

def format_date_for_display(date_str: str) -> str:
    """
    Formats a date string from the Twitter API into a more readable format.
    
    Args:
        date_str: The date string from the API (e.g., "Wed Dec 25 15:43:52 +0000 2019").
        
    Returns:
        A formatted date string (e.g., "25 Dec 2019").
    """
    try:
        dt_object = datetime.strptime(date_str, '%a %b %d %H:%M:%S +0000 %Y')
        return dt_object.strftime('%d %b %Y')
    except (ValueError, TypeError) as e:
        logger.error(f"Error parsing date string '{date_str}': {e}")
        return date_str

def get_selected_communities(community_labels: Dict[str, str], selected_label: str) -> Set[str]:
    """
    Helper function to get the community ID(s) from a selected label string.
    
    Args:
        community_labels: Dictionary mapping community IDs to labels.
        selected_label: The selected label string from a user interface.
        
    Returns:
        A set of community IDs that match the selected label.
    """
    selected_communities = set()
    if selected_label == "All":
        return set(community_labels.keys())
    elif selected_label and selected_label != "Other":
        for comm_id, label in community_labels.items():
            if label == selected_label:
                selected_communities.add(comm_id)
                break
    elif selected_label == "Other":
        # Find all communities that are not in the known labels
        all_ids = set(community_labels.keys())
        known_ids = {comm_id for comm_id, label in community_labels.items()}
        selected_communities = all_ids - known_ids
        
    return selected_communities
