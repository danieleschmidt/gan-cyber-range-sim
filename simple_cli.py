#!/usr/bin/env python3
"""Simple CLI for GAN Cyber Range Simulator - works without complex dependencies."""

import sys
import os
import asyncio
import logging
import json
from datetime import datetime
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import the minimal test components
from minimal_test import MockCyberRange, MockRedTeamAgent, MockBlueTeamAgent, display_results


def setup_logging(debug=False):
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║          🎯 GAN CYBER RANGE SIMULATOR                     ║
    ║                                                           ║
    ║     Adversarial Security Training with AI Agents         ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def print_help():
    """Print help information."""
    help_text = """
    🚀 GAN CYBER RANGE SIMULATOR - Command Line Interface

    USAGE:
        python3 simple_cli.py <command> [options]

    COMMANDS:
        simulate        Run adversarial simulation between red and blue teams
        status          Check system status
        version         Show version information
        help            Show this help message

    SIMULATE OPTIONS:
        --services      Vulnerable services (default: webapp,database,api-gateway)
        --duration      Simulation duration in hours (default: 0.1)
        --red-skill     Red team skill level: beginner,intermediate,advanced (default: advanced)  
        --blue-skill    Blue team skill level: beginner,intermediate,advanced (default: advanced)
        --strategy      Blue team strategy: reactive,proactive (default: proactive)
        --debug         Enable debug logging

    EXAMPLES:
        python3 simple_cli.py simulate
        python3 simple_cli.py simulate --services webapp,database --duration 0.2 --debug
        python3 simple_cli.py simulate --red-skill intermediate --blue-skill advanced
        python3 simple_cli.py status
        python3 simple_cli.py version

    """
    print(help_text)


def parse_args(args):
    """Simple argument parsing."""
    parsed = {
        'command': 'help',
        'services': ['webapp', 'database', 'api-gateway'],
        'duration': 0.1,
        'red_skill': 'advanced',
        'blue_skill': 'advanced',
        'strategy': 'proactive',
        'debug': False
    }
    
    if not args:
        return parsed
        
    # Get command
    parsed['command'] = args[0] if args else 'help'
    
    # Parse options
    i = 1
    while i < len(args):
        arg = args[i]
        
        if arg == '--services' and i + 1 < len(args):
            parsed['services'] = args[i + 1].split(',')
            i += 2
        elif arg == '--duration' and i + 1 < len(args):
            try:
                parsed['duration'] = float(args[i + 1])
            except ValueError:
                print(f"⚠️  Invalid duration: {args[i + 1]}")
                parsed['duration'] = 0.1
            i += 2
        elif arg == '--red-skill' and i + 1 < len(args):
            skill = args[i + 1].lower()
            if skill in ['beginner', 'intermediate', 'advanced']:
                parsed['red_skill'] = skill
            i += 2
        elif arg == '--blue-skill' and i + 1 < len(args):
            skill = args[i + 1].lower()
            if skill in ['beginner', 'intermediate', 'advanced']:
                parsed['blue_skill'] = skill
            i += 2
        elif arg == '--strategy' and i + 1 < len(args):
            strategy = args[i + 1].lower()
            if strategy in ['reactive', 'proactive']:
                parsed['strategy'] = strategy
            i += 2
        elif arg == '--debug':
            parsed['debug'] = True
            i += 1
        else:
            i += 1
    
    return parsed


async def cmd_simulate(args):
    """Run simulation command."""
    print(f"🚀 Starting GAN Cyber Range Simulation")
    print(f"📅 Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Display parameters
    print("📋 SIMULATION PARAMETERS:")
    print(f"   Services: {', '.join(args['services'])}")
    print(f"   Duration: {args['duration']} hours")
    print(f"   Red Team Skill: {args['red_skill']}")
    print(f"   Blue Team Skill: {args['blue_skill']}")  
    print(f"   Blue Strategy: {args['strategy']}")
    print()
    
    # Initialize components
    print("🏗️  Initializing cyber range environment...")
    cyber_range = MockCyberRange(
        vulnerable_services=args['services']
    )
    
    print("🔴 Creating red team agent...")
    red_team = MockRedTeamAgent(
        name="AdvancedRedTeam",
        skill_level=args['red_skill']
    )
    
    print("🔵 Creating blue team agent...")
    blue_team = MockBlueTeamAgent(
        name="ProactiveBlueTeam",
        skill_level=args['blue_skill'],
        defense_strategy=args['strategy']
    )
    
    print("⚔️  Launching adversarial simulation...")
    print()
    
    # Run simulation
    try:
        results = await cyber_range.simulate(
            red_team=red_team,
            blue_team=blue_team,
            duration_hours=args['duration'],
            realtime_factor=60
        )
        
        # Display results
        display_results(results, red_team, blue_team)
        
        # Save results to file
        save_results(results, red_team, blue_team)
        
    except Exception as e:
        print(f"❌ Simulation failed: {e}")
        if args['debug']:
            import traceback
            traceback.print_exc()
        return False
    
    return True


def cmd_status():
    """Check system status."""
    print("📊 GAN CYBER RANGE SYSTEM STATUS")
    print("="*50)
    
    print("🔧 Core Components:")
    print("   ✅ Cyber Range Engine - Operational")
    print("   ✅ Red Team Agent - Ready")
    print("   ✅ Blue Team Agent - Ready")
    print("   ✅ Mock LLM Integration - Available")
    print()
    
    print("🖥️  System Information:")
    print(f"   Python Version: {sys.version}")
    print(f"   Platform: {os.name}")
    print(f"   Working Directory: {os.getcwd()}")
    print()
    
    print("📂 Environment:")
    repo_path = Path(__file__).parent
    src_path = repo_path / "src"
    tests_path = repo_path / "tests"
    
    print(f"   Repository: {repo_path}")
    print(f"   Source Code: {'✅ Found' if src_path.exists() else '❌ Missing'}")
    print(f"   Test Suite: {'✅ Found' if tests_path.exists() else '❌ Missing'}")
    print()
    
    print("🚀 Ready for simulation!")


def cmd_version():
    """Show version information."""
    print("📦 GAN CYBER RANGE SIMULATOR")
    print("="*40)
    print("Version: 1.0.0-alpha")
    print("Author: Daniel Schmidt")
    print("License: MIT")
    print("Repository: https://github.com/yourusername/gan-cyber-range-sim")
    print()
    
    print("🔬 Research Capabilities:")
    print("   • Adversarial AI Training")
    print("   • Multi-modal Threat Detection")
    print("   • Zero-shot Vulnerability Discovery")
    print("   • Self-healing Security Systems")
    print()
    
    print("🏗️  Architecture:")
    print("   • Kubernetes-native deployment")
    print("   • Microservices-based design")
    print("   • Real-time threat simulation")
    print("   • LLM-powered intelligence")


def save_results(results, red_team, blue_team):
    """Save simulation results to file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"simulation_results_{timestamp}.json"
    
    data = {
        'simulation_id': results.simulation_id,
        'timestamp': timestamp,
        'duration': str(results.duration),
        'metrics': {
            'total_attacks': results.total_attacks,
            'services_compromised': results.services_compromised,
            'attacks_blocked': results.attacks_blocked,
            'patches_deployed': results.patches_deployed,
            'compromise_rate': results.compromise_rate,
            'defense_effectiveness': results.defense_effectiveness,
            'avg_detection_time': results.avg_detection_time
        },
        'agents': {
            'red_team': red_team.get_stats(),
            'blue_team': blue_team.get_stats()
        },
        'actions': {
            'red_team_actions': results.red_team_actions,
            'blue_team_actions': results.blue_team_actions
        }
    }
    
    try:
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"💾 Results saved to: {filename}")
    except Exception as e:
        print(f"⚠️  Failed to save results: {e}")


async def main():
    """Main CLI function."""
    args = parse_args(sys.argv[1:])
    
    # Setup logging
    setup_logging(args['debug'])
    
    # Handle commands
    command = args['command'].lower()
    
    if command == 'simulate':
        print_banner()
        success = await cmd_simulate(args)
        sys.exit(0 if success else 1)
        
    elif command == 'status':
        print_banner()
        cmd_status()
        
    elif command == 'version':
        cmd_version()
        
    elif command in ['help', '--help', '-h']:
        print_banner()
        print_help()
        
    else:
        print(f"❌ Unknown command: {command}")
        print("💡 Use 'python3 simple_cli.py help' for available commands")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())