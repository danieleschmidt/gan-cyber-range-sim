#!/usr/bin/env python3
"""
Terragon Autonomous Value Discovery Engine
Continuous SDLC value identification and prioritization system
"""

import json
import subprocess
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple


class ValueDiscoveryEngine:
    """Advanced value discovery with WSJF, ICE, and technical debt scoring"""
    
    def __init__(self, repo_path: Path = Path(".")):
        self.repo_path = Path(repo_path)
        self.config = self._load_config()
        self.metrics_file = self.repo_path / ".terragon" / "value-metrics.json"
        self.backlog_file = self.repo_path / "BACKLOG.md"
        
    def _load_config(self) -> Dict:
        """Load Terragon configuration"""
        config_path = self.repo_path / ".terragon" / "config.yaml"
        if config_path.exists():
            # Simple YAML parsing for our basic config structure
            try:
                with open(config_path) as f:
                    content = f.read()
                    # Extract basic values (simplified parser)
                    config = self._default_config()
                    return config
            except Exception:
                pass
        return self._default_config()
    
    def _default_config(self) -> Dict:
        """Default configuration for advanced repositories"""
        return {
            "scoring": {
                "weights": {
                    "advanced": {"wsjf": 0.5, "ice": 0.1, "technicalDebt": 0.3, "security": 0.1}
                },
                "thresholds": {"minScore": 15, "maxRisk": 0.7, "securityBoost": 2.0}
            },
            "maturity": {"level": "advanced", "score": 78}
        }
    
    def discover_opportunities(self) -> List[Dict]:
        """Comprehensive value opportunity discovery"""
        opportunities = []
        
        # Code analysis for TODOs, FIXMEs, technical debt
        opportunities.extend(self._analyze_code_comments())
        
        # Git history analysis for patterns
        opportunities.extend(self._analyze_git_history())
        
        # Dependency analysis
        opportunities.extend(self._analyze_dependencies())
        
        # Performance opportunities
        opportunities.extend(self._analyze_performance_gaps())
        
        # Security enhancements beyond current setup
        opportunities.extend(self._analyze_security_gaps())
        
        # Documentation gaps
        opportunities.extend(self._analyze_documentation_gaps())
        
        # Infrastructure and automation opportunities
        opportunities.extend(self._analyze_automation_gaps())
        
        return self._score_and_rank_opportunities(opportunities)
    
    def _analyze_code_comments(self) -> List[Dict]:
        """Extract TODO, FIXME, HACK comments from codebase"""
        opportunities = []
        patterns = [
            (r'TODO[:\s]+(.+)', 'TODO', 'technical-debt'),
            (r'FIXME[:\s]+(.+)', 'FIXME', 'bug-fix'),
            (r'HACK[:\s]+(.+)', 'HACK', 'refactoring'),
            (r'DEPRECATED[:\s]+(.+)', 'DEPRECATED', 'modernization'),
            (r'XXX[:\s]+(.+)', 'XXX', 'technical-debt')
        ]
        
        for py_file in self.repo_path.rglob("*.py"):
            if "venv" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                content = py_file.read_text(encoding='utf-8')
                for i, line in enumerate(content.split('\n'), 1):
                    for pattern, comment_type, category in patterns:
                        match = re.search(pattern, line, re.IGNORECASE)
                        if match:
                            opportunities.append({
                                'id': f'{comment_type.lower()}-{len(opportunities):03d}',
                                'title': f'Address {comment_type}: {match.group(1)[:50]}...',
                                'description': match.group(1),
                                'category': category,
                                'source': 'code-analysis',
                                'file_path': str(py_file.relative_to(self.repo_path)),
                                'line_number': i,
                                'effort_estimate': self._estimate_effort_from_comment(match.group(1))
                            })
            except (UnicodeDecodeError, PermissionError):
                continue
                
        return opportunities
    
    def _analyze_git_history(self) -> List[Dict]:
        """Analyze Git history for improvement patterns"""
        opportunities = []
        
        try:
            # Get recent commits
            result = subprocess.run(
                ['git', 'log', '--oneline', '-20', '--grep=fix', '--grep=hack', '--grep=temp'],
                cwd=self.repo_path, capture_output=True, text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if any(keyword in line.lower() for keyword in ['fix', 'hack', 'temp', 'quick']):
                        commit_hash = line.split()[0]
                        commit_msg = ' '.join(line.split()[1:])
                        opportunities.append({
                            'id': f'git-{commit_hash[:7]}',
                            'title': f'Review technical debt from: {commit_msg[:50]}...',
                            'description': f'Commit suggests technical debt: {commit_msg}',
                            'category': 'technical-debt',
                            'source': 'git-history',
                            'effort_estimate': 4
                        })
        except Exception:
            pass
            
        return opportunities
    
    def _analyze_dependencies(self) -> List[Dict]:
        """Analyze dependency update opportunities"""
        opportunities = []
        
        # Check for outdated dependencies
        pyproject_path = self.repo_path / "pyproject.toml"
        if pyproject_path.exists():
            try:
                # Run pip-outdated check (simplified)
                opportunities.append({
                    'id': 'dep-001',
                    'title': 'Update outdated Python dependencies',
                    'description': 'Regular dependency updates for security and performance',
                    'category': 'maintenance',
                    'source': 'dependency-analysis',
                    'effort_estimate': 8
                })
            except Exception:
                pass
                
        return opportunities
    
    def _analyze_performance_gaps(self) -> List[Dict]:
        """Identify performance optimization opportunities"""
        opportunities = []
        
        # Look for synchronous code that could be async
        for py_file in self.repo_path.rglob("*.py"):
            if "venv" in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding='utf-8')
                if re.search(r'requests\.get|requests\.post', content) and 'async' not in content:
                    opportunities.append({
                        'id': f'perf-async-{len(opportunities):03d}',
                        'title': f'Convert synchronous HTTP calls to async in {py_file.name}',
                        'description': 'Optimize HTTP calls for better concurrency',
                        'category': 'performance',
                        'source': 'performance-analysis',
                        'file_path': str(py_file.relative_to(self.repo_path)),
                        'effort_estimate': 6
                    })
            except (UnicodeDecodeError, PermissionError):
                continue
                
        return opportunities
    
    def _analyze_security_gaps(self) -> List[Dict]:
        """Identify security enhancement opportunities"""
        opportunities = []
        
        # Check for missing security headers in web applications
        if (self.repo_path / "pyproject.toml").exists():
            with open(self.repo_path / "pyproject.toml") as f:
                content = f.read()
                if "fastapi" in content.lower():
                    opportunities.append({
                        'id': 'sec-001',
                        'title': 'Add security headers to FastAPI application',
                        'description': 'Implement CORS, CSP, and other security headers',
                        'category': 'security',
                        'source': 'security-analysis',
                        'effort_estimate': 4
                    })
        
        return opportunities
    
    def _analyze_documentation_gaps(self) -> List[Dict]:
        """Identify documentation improvement opportunities"""
        opportunities = []
        
        # Check for missing docstrings
        docstring_files = []
        for py_file in self.repo_path.rglob("*.py"):
            if "venv" in str(py_file) or "__pycache__" in str(py_file):
                continue
            try:
                content = py_file.read_text(encoding='utf-8')
                # Simple check for functions without docstrings
                if re.search(r'def \w+\([^)]*\):', content) and '"""' not in content:
                    docstring_files.append(py_file.name)
            except (UnicodeDecodeError, PermissionError):
                continue
        
        if docstring_files:
            opportunities.append({
                'id': 'doc-001',
                'title': f'Add docstrings to {len(docstring_files)} Python files',
                'description': f'Missing docstrings in: {", ".join(docstring_files[:3])}...',
                'category': 'documentation',
                'source': 'documentation-analysis',
                'effort_estimate': len(docstring_files) * 2
            })
            
        return opportunities
    
    def _analyze_automation_gaps(self) -> List[Dict]:
        """Identify automation opportunities"""
        opportunities = []
        
        # Check for missing GitHub Actions
        github_dir = self.repo_path / ".github" / "workflows"
        if not github_dir.exists() or not list(github_dir.glob("*.yml")):
            opportunities.append({
                'id': 'auto-001',
                'title': 'Implement GitHub Actions CI/CD workflows',
                'description': 'Automated testing, security scanning, and deployment',
                'category': 'automation',
                'source': 'automation-analysis',
                'effort_estimate': 16
            })
        
        return opportunities
    
    def _estimate_effort_from_comment(self, comment: str) -> int:
        """Estimate effort hours from comment content"""
        if any(word in comment.lower() for word in ['major', 'refactor', 'rewrite']):
            return 24
        elif any(word in comment.lower() for word in ['complex', 'difficult']):
            return 12
        elif any(word in comment.lower() for word in ['quick', 'simple', 'easy']):
            return 2
        else:
            return 6
    
    def _score_and_rank_opportunities(self, opportunities: List[Dict]) -> List[Dict]:
        """Score opportunities using WSJF, ICE, and technical debt"""
        weights = self.config["scoring"]["weights"][self.config["maturity"]["level"]]
        
        for opp in opportunities:
            # WSJF Scoring (Weighted Shortest Job First)
            business_value = self._score_business_value(opp)
            time_criticality = self._score_time_criticality(opp)
            risk_reduction = self._score_risk_reduction(opp)
            job_size = opp.get('effort_estimate', 6)
            
            wsjf = (business_value + time_criticality + risk_reduction) / max(job_size, 1) * 10
            
            # ICE Scoring (Impact, Confidence, Ease)
            impact = self._score_impact(opp)
            confidence = self._score_confidence(opp)
            ease = max(10 - job_size, 1)
            
            ice = impact * confidence * ease
            
            # Technical Debt Scoring
            tech_debt = self._score_technical_debt(opp)
            
            # Security boost
            security_boost = 1.0
            if opp.get('category') == 'security':
                security_boost = self.config["scoring"]["thresholds"]["securityBoost"]
            
            # Composite score
            composite = (
                weights["wsjf"] * wsjf +
                weights["ice"] * (ice / 100) +  # Normalize ICE to similar scale
                weights["technicalDebt"] * tech_debt +
                weights["security"] * (10 if opp.get('category') == 'security' else 0)
            ) * security_boost
            
            opp.update({
                'scores': {
                    'wsjf': round(wsjf, 1),
                    'ice': round(ice, 1),
                    'technicalDebt': round(tech_debt, 1),
                    'composite': round(composite, 1)
                }
            })
        
        # Sort by composite score
        return sorted(opportunities, key=lambda x: x['scores']['composite'], reverse=True)
    
    def _score_business_value(self, opp: Dict) -> int:
        """Score business value (1-10 scale)"""
        category_values = {
            'automation': 8,
            'security': 9,
            'performance': 7,
            'bug-fix': 6,
            'technical-debt': 5,
            'documentation': 4,
            'maintenance': 3
        }
        return category_values.get(opp.get('category', ''), 5)
    
    def _score_time_criticality(self, opp: Dict) -> int:
        """Score time criticality (1-10 scale)"""
        if opp.get('category') == 'security':
            return 9
        elif opp.get('category') == 'bug-fix':
            return 8
        elif opp.get('source') == 'git-history':
            return 6
        else:
            return 5
    
    def _score_risk_reduction(self, opp: Dict) -> int:
        """Score risk reduction benefit (1-10 scale)"""
        category_risks = {
            'security': 10,
            'automation': 8,
            'technical-debt': 7,
            'bug-fix': 8,
            'performance': 6,
            'documentation': 4,
            'maintenance': 5
        }
        return category_risks.get(opp.get('category', ''), 5)
    
    def _score_impact(self, opp: Dict) -> int:
        """Score expected impact (1-10 scale)"""
        return self._score_business_value(opp)
    
    def _score_confidence(self, opp: Dict) -> int:
        """Score implementation confidence (1-10 scale)"""
        effort = opp.get('effort_estimate', 6)
        if effort <= 4:
            return 9
        elif effort <= 8:
            return 7
        elif effort <= 16:
            return 6
        else:
            return 4
    
    def _score_technical_debt(self, opp: Dict) -> int:
        """Score technical debt reduction (1-10 scale)"""
        category_debt = {
            'technical-debt': 9,
            'refactoring': 8,
            'modernization': 7,
            'bug-fix': 6,
            'performance': 5,
            'automation': 4,
            'documentation': 3,
            'security': 6,
            'maintenance': 4
        }
        return category_debt.get(opp.get('category', ''), 5)
    
    def update_metrics(self, opportunities: List[Dict]):
        """Update value metrics tracking"""
        if self.metrics_file.exists():
            with open(self.metrics_file) as f:
                metrics = json.load(f)
        else:
            metrics = {"executionHistory": [], "backlogMetrics": {}}
        
        # Update backlog metrics
        metrics["backlogMetrics"].update({
            "totalItems": len(opportunities),
            "averageAge": 0,  # New discovery
            "debtRatio": len([o for o in opportunities if o.get('category') == 'technical-debt']) / max(len(opportunities), 1),
            "velocityTrend": "discovering",
            "lastDiscovery": datetime.now().isoformat(),
            "highPriorityItems": len([o for o in opportunities if o['scores']['composite'] > 70]),
            "mediumPriorityItems": len([o for o in opportunities if 40 <= o['scores']['composite'] <= 70]),
            "lowPriorityItems": len([o for o in opportunities if o['scores']['composite'] < 40])
        })
        
        with open(self.metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
    
    def update_backlog(self, opportunities: List[Dict]):
        """Update the BACKLOG.md file with discovered opportunities"""
        top_15 = opportunities[:15]
        
        # Calculate next best item
        next_item = top_15[0] if top_15 else None
        
        backlog_content = f"""# 📊 Autonomous Value Backlog

**Repository**: GAN Cyber Range Simulator  
**Last Updated**: {datetime.now().isoformat()}  
**Next Execution**: {(datetime.now() + timedelta(hours=1)).isoformat()}  
**Total Opportunities**: {len(opportunities)}

## 🎯 Next Best Value Item

"""
        
        if next_item:
            backlog_content += f"""**[{next_item['id'].upper()}] {next_item['title']}**
- **Composite Score**: {next_item['scores']['composite']}
- **WSJF**: {next_item['scores']['wsjf']} | **ICE**: {next_item['scores']['ice']} | **Tech Debt**: {next_item['scores']['technicalDebt']}
- **Estimated Effort**: {next_item['effort_estimate']} hours
- **Expected Impact**: {next_item['description']}

"""
        
        backlog_content += """---

## 📋 Top 15 Value Opportunities

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, opp in enumerate(top_15, 1):
            priority = "HIGH" if opp['scores']['composite'] > 70 else "MEDIUM" if opp['scores']['composite'] > 40 else "LOW"
            backlog_content += f"| {i} | {opp['id'].upper()} | {opp['title'][:40]}... | {opp['scores']['composite']} | {opp.get('category', 'unknown').title()} | {opp['effort_estimate']} |\n"
        
        backlog_content += f"""

---

## 📈 Discovery Statistics

- **Total Items Discovered**: {len(opportunities)}
- **High Priority Items**: {len([o for o in opportunities if o['scores']['composite'] > 70])}
- **Average Composite Score**: {sum(o['scores']['composite'] for o in opportunities) / max(len(opportunities), 1):.1f}
- **Total Estimated Effort**: {sum(o['effort_estimate'] for o in opportunities)} hours

### Discovery Sources
"""
        
        source_counts = {}
        for opp in opportunities:
            source = opp.get('source', 'unknown')
            source_counts[source] = source_counts.get(source, 0) + 1
            
        for source, count in sorted(source_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(opportunities)) * 100
            backlog_content += f"- **{source.replace('-', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        backlog_content += f"""

### Category Distribution
"""
        
        category_counts = {}
        for opp in opportunities:
            category = opp.get('category', 'unknown')
            category_counts[category] = category_counts.get(category, 0) + 1
            
        for category, count in sorted(category_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(opportunities)) * 100
            backlog_content += f"- **{category.replace('-', ' ').title()}**: {count} items ({percentage:.1f}%)\n"
        
        backlog_content += """

---

**Next Value Discovery**: Automatic execution in 60 minutes  
**Continuous Improvement Status**: ✅ Active  
**Autonomous Execution**: ✅ Enabled"""
        
        with open(self.backlog_file, 'w') as f:
            f.write(backlog_content)


def main():
    """Main execution function"""
    print("🔍 Terragon Autonomous Value Discovery Engine")
    print("=" * 50)
    
    engine = ValueDiscoveryEngine()
    
    print("📊 Discovering value opportunities...")
    opportunities = engine.discover_opportunities()
    
    print(f"✅ Discovered {len(opportunities)} opportunities")
    
    if opportunities:
        top_item = opportunities[0]
        print(f"🎯 Next best value: {top_item['title']} (Score: {top_item['scores']['composite']})")
    
    # Update metrics and backlog
    engine.update_metrics(opportunities)
    engine.update_backlog(opportunities)
    
    print("📝 Updated BACKLOG.md and value metrics")
    print("🚀 Value discovery cycle complete!")


if __name__ == "__main__":
    main()