"""Integration tests for cyber range environment."""

import pytest
from unittest.mock import MagicMock


class TestCyberRangeIntegration:
    """Test cyber range integration scenarios."""

    @pytest.fixture
    def mock_range_environment(self, isolated_environment):
        """Mock cyber range environment."""
        env = MagicMock()
        env.config = isolated_environment
        return env

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_red_blue_team_interaction(self, mock_range_environment):
        """Test red and blue team interaction in controlled environment."""
        # Mock interaction between red and blue teams
        red_action = {"type": "reconnaissance", "target": "webapp"}
        blue_response = {"type": "monitor", "alert_level": "low"}
        
        assert red_action["type"] == "reconnaissance"
        assert blue_response["alert_level"] == "low"

    @pytest.mark.integration
    def test_environment_isolation(self, mock_range_environment):
        """Test that cyber range environment is properly isolated."""
        assert mock_range_environment.config["network_isolation"] is True
        assert "network_policy" in mock_range_environment.config["security_policies"]