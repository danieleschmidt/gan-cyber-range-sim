"""Command line interface for GAN Cyber Range Simulator."""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .environment.cyber_range import CyberRange
from .agents.red_team import RedTeamAgent
from .agents.blue_team import BlueTeamAgent


console = Console()


def setup_logging(debug: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


@click.group()
@click.option("--debug", is_flag=True, help="Enable debug logging")
@click.pass_context
def main(ctx: click.Context, debug: bool) -> None:
    """GAN Cyber Range Simulator - Adversarial security training platform."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug
    setup_logging(debug)


@main.command()
@click.option("--services", "-s", multiple=True, default=["webapp", "database", "api-gateway"],
              help="Vulnerable services to deploy")
@click.option("--topology", "-t", default="multi-tier", help="Network topology")
@click.option("--difficulty", "-d", default="medium", help="Simulation difficulty")
@click.option("--duration", default=1.0, help="Simulation duration in hours")
@click.option("--realtime-factor", default=60, help="Time acceleration factor")
@click.option("--red-model", default="gpt-4", help="Red team LLM model")
@click.option("--blue-model", default="claude-3", help="Blue team LLM model")
@click.option("--red-skill", default="advanced", help="Red team skill level")
@click.option("--blue-skill", default="advanced", help="Blue team skill level")
def simulate(
    services: tuple,
    topology: str,
    difficulty: str,
    duration: float,
    realtime_factor: int,
    red_model: str,
    blue_model: str,
    red_skill: str,
    blue_skill: str
) -> None:
    """Run adversarial simulation between red and blue teams."""
    console.print("[bold green]🎯 Starting GAN Cyber Range Simulation[/bold green]")
    
    # Display simulation parameters
    params_table = Table(title="Simulation Parameters")
    params_table.add_column("Parameter", style="cyan")
    params_table.add_column("Value", style="green")
    
    params_table.add_row("Services", ", ".join(services))
    params_table.add_row("Topology", topology)
    params_table.add_row("Difficulty", difficulty)
    params_table.add_row("Duration", f"{duration} hours")
    params_table.add_row("Time Factor", f"{realtime_factor}x")
    params_table.add_row("Red Team Model", red_model)
    params_table.add_row("Blue Team Model", blue_model)
    
    console.print(params_table)
    
    # Initialize environment
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        init_task = progress.add_task("Initializing cyber range...", total=None)
        
        # Create cyber range
        cyber_range = CyberRange(
            vulnerable_services=list(services),
            network_topology=topology,
            difficulty=difficulty
        )
        
        progress.update(init_task, description="Creating red team agent...")
        red_team = RedTeamAgent(
            name="AdvancedRedTeam",
            llm_model=red_model,
            skill_level=red_skill
        )
        
        progress.update(init_task, description="Creating blue team agent...")
        blue_team = BlueTeamAgent(
            name="ProactiveBlueTeam",
            llm_model=blue_model,
            skill_level=blue_skill,
            defense_strategy="proactive"
        )
        
        progress.update(init_task, description="Starting simulation...")
        progress.remove_task(init_task)
    
    # Run simulation
    try:
        results = asyncio.run(
            cyber_range.simulate(
                red_team=red_team,
                blue_team=blue_team,
                duration_hours=duration,
                realtime_factor=realtime_factor
            )
        )
        
        # Display results
        _display_simulation_results(results, red_team, blue_team)
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Simulation interrupted by user[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[red]Simulation failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--namespace", "-n", default="cyber-range", help="Kubernetes namespace")
@click.option("--services", "-s", multiple=True, default=["webapp", "database"],
              help="Services to deploy")
def deploy(namespace: str, services: tuple) -> None:
    """Deploy cyber range to Kubernetes cluster."""
    console.print("[bold blue]🚀 Deploying Cyber Range to Kubernetes[/bold blue]")
    
    try:
        from .environment.kubernetes_manager import KubernetesManager
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            deploy_task = progress.add_task("Connecting to Kubernetes...", total=None)
            
            k8s_manager = KubernetesManager(namespace=namespace)
            
            progress.update(deploy_task, description="Creating namespace...")
            asyncio.run(k8s_manager.create_namespace())
            
            progress.update(deploy_task, description="Setting up network policies...")
            asyncio.run(k8s_manager.create_network_policies())
            
            for service in services:
                progress.update(deploy_task, description=f"Deploying {service}...")
                service_config = {
                    "name": service,
                    "image": f"gan-cyber-range/{service}:latest",
                    "ports": [80, 443] if service == "webapp" else [3306]
                }
                asyncio.run(k8s_manager.deploy_vulnerable_service(service_config))
            
            progress.update(deploy_task, description="Deploying monitoring stack...")
            asyncio.run(k8s_manager.deploy_monitoring_stack())
            
        console.print(f"[green]✅ Deployment completed in namespace: {namespace}[/green]")
        
    except ImportError:
        console.print("[red]Kubernetes client not available. Install with: pip install kubernetes[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Deployment failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--namespace", "-n", default="cyber-range", help="Kubernetes namespace")
def status(namespace: str) -> None:
    """Check status of deployed cyber range."""
    console.print("[bold cyan]📊 Cyber Range Status[/bold cyan]")
    
    try:
        from .environment.kubernetes_manager import KubernetesManager
        
        k8s_manager = KubernetesManager(namespace=namespace)
        cluster_info = asyncio.run(k8s_manager.get_cluster_info())
        
        # Display cluster information
        status_table = Table(title="Cluster Status")
        status_table.add_column("Component", style="cyan")
        status_table.add_column("Status", style="green")
        
        status_table.add_row("Cluster", "Healthy" if cluster_info.get("cluster_healthy") else "Unhealthy")
        status_table.add_row("Namespace", cluster_info.get("namespace", "Unknown"))
        status_table.add_row("Nodes", str(cluster_info.get("nodes", 0)))
        
        console.print(status_table)
        
    except ImportError:
        console.print("[red]Kubernetes client not available[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Status check failed: {e}[/red]")
        sys.exit(1)


@main.command()
@click.option("--namespace", "-n", default="cyber-range", help="Kubernetes namespace")
@click.confirmation_option(prompt="Are you sure you want to cleanup the cyber range?")
def cleanup(namespace: str) -> None:
    """Clean up deployed cyber range resources."""
    console.print("[bold red]🧹 Cleaning up Cyber Range[/bold red]")
    
    try:
        from .environment.kubernetes_manager import KubernetesManager
        
        k8s_manager = KubernetesManager(namespace=namespace)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console
        ) as progress:
            cleanup_task = progress.add_task("Cleaning up resources...", total=None)
            
            asyncio.run(k8s_manager.cleanup_namespace())
            
        console.print("[green]✅ Cleanup completed[/green]")
        
    except ImportError:
        console.print("[red]Kubernetes client not available[/red]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Cleanup failed: {e}[/red]")
        sys.exit(1)


@main.command()
def version() -> None:
    """Show version information."""
    from . import __version__, __author__
    console.print(f"[bold green]GAN Cyber Range Simulator v{__version__}[/bold green]")
    console.print(f"Author: {__author__}")


def _display_simulation_results(results, red_team, blue_team) -> None:
    """Display simulation results in a formatted table."""
    console.print("\n[bold green]🎉 Simulation Complete![/bold green]")
    
    # Summary table
    summary_table = Table(title="Simulation Results")
    summary_table.add_column("Metric", style="cyan")
    summary_table.add_column("Value", style="green")
    
    summary_table.add_row("Duration", str(results.duration))
    summary_table.add_row("Total Attacks", str(results.total_attacks))
    summary_table.add_row("Services Compromised", str(results.services_compromised))
    summary_table.add_row("Attacks Blocked", str(results.attacks_blocked))
    summary_table.add_row("Patches Deployed", str(results.patches_deployed))
    summary_table.add_row("Compromise Rate", f"{results.compromise_rate:.2%}")
    summary_table.add_row("Defense Effectiveness", f"{results.defense_effectiveness:.2%}")
    summary_table.add_row("Avg Detection Time", results.avg_detection_time)
    
    console.print(summary_table)
    
    # Agent statistics
    agent_table = Table(title="Agent Performance")
    agent_table.add_column("Agent", style="cyan")
    agent_table.add_column("Actions", style="green")
    agent_table.add_column("Success Rate", style="green")
    agent_table.add_column("Skill Level", style="yellow")
    
    red_stats = red_team.get_stats()
    blue_stats = blue_team.get_stats()
    
    agent_table.add_row(
        red_stats["name"],
        str(red_stats["total_actions"]),
        f"{red_stats['success_rate']:.2%}",
        red_stats["skill_level"]
    )
    
    agent_table.add_row(
        blue_stats["name"],
        str(blue_stats["total_actions"]),
        f"{blue_stats['success_rate']:.2%}",
        blue_stats["skill_level"]
    )
    
    console.print(agent_table)


if __name__ == "__main__":
    main()