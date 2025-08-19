#!/usr/bin/env python3
"""Autonomous quality pipeline runner for GAN Cyber Range."""

import asyncio
import sys
import argparse
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from gan_cyber_range.quality import (
    AutomatedQualityPipeline,
    ProgressiveValidator,
    ValidationStage
)


def setup_logging(level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler("quality_pipeline.log")
        ]
    )


async def console_notification_handler(notification_data):
    """Console notification handler."""
    print(f"\n🔔 PIPELINE NOTIFICATION:")
    print(f"   Pipeline ID: {notification_data['pipeline_id']}")
    print(f"   Status: {notification_data['status']}")
    print(f"   Score: {notification_data['overall_score']:.1f}%")
    print(f"   Deployment Ready: {notification_data['deployment_ready']}")
    print(f"   Summary: {notification_data['summary']}\n")


async def main():
    """Main pipeline execution."""
    parser = argparse.ArgumentParser(description="Run autonomous quality pipeline")
    
    parser.add_argument(
        "--target-stage",
        choices=["generation_1", "generation_2", "generation_3", "production"],
        default="production",
        help="Target validation stage"
    )
    
    parser.add_argument(
        "--config",
        type=str,
        help="Configuration file path"
    )
    
    parser.add_argument(
        "--project-root",
        type=str,
        default=".",
        help="Project root directory"
    )
    
    parser.add_argument(
        "--auto-deploy",
        action="store_true",
        help="Enable automatic deployment if pipeline passes"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    parser.add_argument(
        "--trigger",
        type=str,
        default="manual",
        help="Pipeline trigger event"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger("quality_pipeline_runner")
    
    try:
        # Initialize pipeline
        pipeline = AutomatedQualityPipeline(
            project_root=args.project_root,
            config_file=args.config,
            auto_deploy=args.auto_deploy
        )
        
        # Register console notification handler
        pipeline.register_notification_handler(console_notification_handler)
        
        # Parse target stage
        target_stage = ValidationStage(args.target_stage)
        
        print(f"🚀 Starting Autonomous Quality Pipeline")
        print(f"   Project Root: {Path(args.project_root).absolute()}")
        print(f"   Target Stage: {target_stage.value}")
        print(f"   Auto Deploy: {args.auto_deploy}")
        print(f"   Trigger: {args.trigger}")
        print()
        
        # Run pipeline
        result = await pipeline.run_pipeline(
            trigger_event=args.trigger,
            target_stage=target_stage
        )
        
        # Display results
        print(f"\n{'='*60}")
        print(f"🎯 PIPELINE EXECUTION COMPLETE")
        print(f"{'='*60}")
        print(f"Pipeline ID: {result.pipeline_id}")
        print(f"Status: {result.status.value.upper()}")
        print(f"Overall Score: {result.overall_score:.1f}%")
        print(f"Success Rate: {result.success_rate:.1f}%")
        print(f"Execution Time: {result.total_execution_time:.1f}s")
        print(f"Deployment Ready: {result.deployment_ready}")
        print(f"Stages Completed: {len(result.stages_completed)}")
        
        if result.validation_results:
            print(f"\n📊 VALIDATION RESULTS:")
            for vr in result.validation_results:
                print(f"   {vr.stage.value}: {vr.status.value} ({vr.metrics.overall_score:.1f}%)")
        
        if result.artifacts:
            print(f"\n📁 ARTIFACTS GENERATED:")
            for artifact in result.artifacts[:10]:  # Show first 10
                print(f"   {artifact}")
            if len(result.artifacts) > 10:
                print(f"   ... and {len(result.artifacts) - 10} more")
        
        if result.error_messages:
            print(f"\n❌ ERRORS:")
            for error in result.error_messages:
                print(f"   {error}")
        
        print(f"\n{'='*60}")
        
        # Exit with appropriate code
        if result.status.value in ["passed", "warning"]:
            if result.deployment_ready:
                print("✅ Pipeline passed and deployment ready!")
                sys.exit(0)
            else:
                print("⚠️  Pipeline passed but deployment not ready")
                sys.exit(1)
        else:
            print("❌ Pipeline failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Pipeline execution interrupted by user")
        print("\n⚠️  Pipeline execution interrupted")
        sys.exit(130)
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        print(f"\n❌ Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())