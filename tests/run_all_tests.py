"""
Complete Test Suite Runner for Namel3ss Parallel and Distributed Execution.

This master test runner executes all testing suites including:
- Unit tests
- Integration tests  
- Performance and benchmarking tests
- Edge case and stress tests
- Fault tolerance tests
- Security validation tests

Provides comprehensive production deployment validation.
"""

import asyncio
import json
import logging
import time
import os
import sys
from typing import Dict, Any, List

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all test suites
from test_comprehensive_suite import ComprehensiveTestRunner
from test_fault_tolerance import FaultToleranceTestRunner
from test_security_validation import SecurityTestRunner

logger = logging.getLogger(__name__)


class MasterTestRunner:
    """Master test runner for complete system validation."""
    
    def __init__(self):
        self.all_results = {}
        self.start_time = None
        self.end_time = None
        self.test_suites = {
            'comprehensive': ComprehensiveTestRunner(),
            'fault_tolerance': FaultToleranceTestRunner(), 
            'security': SecurityTestRunner()
        }
    
    async def run_complete_test_suite(self, quick_mode: bool = False) -> Dict[str, Any]:
        """
        Run complete test suite across all categories.
        
        Args:
            quick_mode: If True, skip some performance and stress tests
        """
        print("🚀 Starting COMPLETE Namel3ss Test Suite Validation...\n")
        print("=" * 80)
        print("🎯 OBJECTIVE: Comprehensive Production Deployment Validation")
        print("🔬 SCOPE: Unit, Integration, Performance, Fault Tolerance, Security")
        print("⚡ MODE:", "Quick Validation" if quick_mode else "Full Comprehensive")
        print("=" * 80)
        print()
        
        self.start_time = time.time()
        
        # 1. Comprehensive Suite (Unit, Integration, Performance, Edge Cases)
        print("📋 PHASE 1: Comprehensive Testing Suite")
        print("─" * 50)
        comprehensive_report = await self.test_suites['comprehensive'].run_all_tests(
            include_performance=not quick_mode,
            include_edge_cases=True
        )
        self.all_results['comprehensive'] = comprehensive_report
        self._print_phase_summary("Comprehensive", comprehensive_report)
        print()
        
        # 2. Fault Tolerance Suite 
        print("🛡️ PHASE 2: Fault Tolerance & Recovery Testing")
        print("─" * 50)
        fault_tolerance_report = await self.test_suites['fault_tolerance'].run_all_fault_tolerance_tests()
        self.all_results['fault_tolerance'] = fault_tolerance_report
        self._print_phase_summary("Fault Tolerance", fault_tolerance_report)
        print()
        
        # 3. Security Validation Suite
        print("🔐 PHASE 3: Security Validation Testing")
        print("─" * 50)
        security_report = await self.test_suites['security'].run_all_security_tests()
        self.all_results['security'] = security_report
        self._print_phase_summary("Security", security_report)
        print()
        
        self.end_time = time.time()
        
        # Generate comprehensive report
        return self._generate_master_report()
    
    def _print_phase_summary(self, phase_name: str, report: Dict[str, Any]):
        """Print summary for a test phase."""
        if 'summary' in report:
            summary = report['summary']
            passed = summary['passed_tests']
            total = summary['total_tests']
            rate = summary['success_rate']
            duration = summary['total_duration_seconds']
        elif 'comprehensive_summary' in report:
            summary = report['comprehensive_summary']
            passed = summary['passed_tests']
            total = summary['total_tests'] 
            rate = summary['success_rate']
            duration = summary['total_duration_seconds']
        elif 'fault_tolerance_summary' in report:
            summary = report['fault_tolerance_summary']
            passed = summary['passed_tests']
            total = summary['total_tests']
            rate = summary['success_rate']
            duration = summary['total_duration_seconds']
        elif 'security_summary' in report:
            summary = report['security_summary']
            passed = summary['passed_tests']
            total = summary['total_tests']
            rate = summary['success_rate']
            duration = summary['total_duration_seconds']
        else:
            print(f"  ⚠️ {phase_name}: Report format not recognized")
            return
        
        status_icon = "✅" if rate >= 95 else "⚠️" if rate >= 80 else "❌"
        print(f"  {status_icon} {phase_name}: {passed}/{total} tests passed ({rate:.1f}%) - {duration:.1f}s")
    
    def _generate_master_report(self) -> Dict[str, Any]:
        """Generate comprehensive master test report."""
        total_duration = self.end_time - self.start_time
        
        # Aggregate statistics across all test suites
        aggregate_stats = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'total_duration_seconds': total_duration
        }
        
        # Extract stats from each suite
        for suite_name, report in self.all_results.items():
            if 'summary' in report:
                summary = report['summary']
            elif 'comprehensive_summary' in report:
                summary = report['comprehensive_summary']
            elif 'fault_tolerance_summary' in report:
                summary = report['fault_tolerance_summary']
            elif 'security_summary' in report:
                summary = report['security_summary']
            else:
                continue
                
            aggregate_stats['total_tests'] += summary['total_tests']
            aggregate_stats['passed_tests'] += summary['passed_tests']
            aggregate_stats['failed_tests'] += summary['failed_tests']
        
        # Calculate overall success rate
        overall_success_rate = (
            aggregate_stats['passed_tests'] / aggregate_stats['total_tests'] * 100
            if aggregate_stats['total_tests'] > 0 else 0
        )
        
        # Determine production readiness
        production_readiness_criteria = {
            'comprehensive_tests_passed': self._get_success_rate('comprehensive') >= 90,
            'fault_tolerance_validated': self._get_success_rate('fault_tolerance') >= 85, 
            'security_validated': self._get_success_rate('security') >= 95,
            'overall_success_rate': overall_success_rate >= 90
        }
        
        production_ready = all(production_readiness_criteria.values())
        
        # Generate deployment recommendations
        deployment_recommendations = self._generate_deployment_recommendations(production_readiness_criteria)
        
        # Create master report
        master_report = {
            'execution_summary': {
                'execution_timestamp': time.time(),
                'total_duration_seconds': total_duration,
                'test_suites_executed': list(self.test_suites.keys()),
                'execution_mode': 'comprehensive'
            },
            'aggregate_statistics': {
                **aggregate_stats,
                'overall_success_rate': overall_success_rate
            },
            'production_readiness': {
                'criteria': production_readiness_criteria,
                'ready_for_production': production_ready,
                'confidence_score': self._calculate_confidence_score()
            },
            'deployment_recommendations': deployment_recommendations,
            'detailed_results': self.all_results,
            'quality_metrics': self._calculate_quality_metrics()
        }
        
        # Print comprehensive summary
        self._print_master_summary(master_report)
        
        return master_report
    
    def _get_success_rate(self, suite_name: str) -> float:
        """Get success rate for a specific test suite."""
        report = self.all_results.get(suite_name, {})
        
        if 'summary' in report:
            return report['summary']['success_rate']
        elif 'comprehensive_summary' in report:
            return report['comprehensive_summary']['success_rate']
        elif 'fault_tolerance_summary' in report:
            return report['fault_tolerance_summary']['success_rate']
        elif 'security_summary' in report:
            return report['security_summary']['success_rate']
        
        return 0.0
    
    def _generate_deployment_recommendations(self, criteria: Dict[str, bool]) -> List[str]:
        """Generate deployment recommendations based on test results."""
        recommendations = []
        
        if criteria['overall_success_rate'] and criteria['comprehensive_tests_passed'] and \
           criteria['fault_tolerance_validated'] and criteria['security_validated']:
            recommendations.append("✅ RECOMMENDED: Deploy to production with confidence")
            recommendations.append("✅ All critical systems validated and production-ready")
            recommendations.append("✅ Monitoring and observability systems operational")
        
        elif criteria['comprehensive_tests_passed'] and criteria['security_validated']:
            recommendations.append("⚠️ CONDITIONAL: Deploy with enhanced monitoring")
            if not criteria['fault_tolerance_validated']:
                recommendations.append("⚠️ Monitor fault tolerance closely in production")
                recommendations.append("⚠️ Consider staged rollout with fallback capabilities")
        
        elif criteria['security_validated']:
            recommendations.append("❌ NOT RECOMMENDED: Address core functionality issues")
            recommendations.append("❌ Complete comprehensive testing before deployment")
            if not criteria['fault_tolerance_validated']:
                recommendations.append("❌ Implement proper fault tolerance mechanisms")
        
        else:
            recommendations.append("❌ CRITICAL: Do not deploy to production")
            recommendations.append("❌ Address all security vulnerabilities immediately")
            recommendations.append("❌ Complete full test validation before considering deployment")
        
        # Add specific recommendations based on individual suite performance
        comprehensive_rate = self._get_success_rate('comprehensive')
        fault_rate = self._get_success_rate('fault_tolerance')
        security_rate = self._get_success_rate('security')
        
        if comprehensive_rate < 85:
            recommendations.append("🔧 Focus on core functionality and integration issues")
        
        if fault_rate < 80:
            recommendations.append("🛡️ Strengthen fault tolerance and recovery mechanisms")
        
        if security_rate < 95:
            recommendations.append("🔐 Address security vulnerabilities before any deployment")
        
        return recommendations
    
    def _calculate_confidence_score(self) -> float:
        """Calculate overall confidence score for production deployment."""
        weights = {
            'comprehensive': 0.4,  # 40% weight
            'fault_tolerance': 0.3,  # 30% weight
            'security': 0.3  # 30% weight
        }
        
        weighted_score = 0
        for suite_name, weight in weights.items():
            success_rate = self._get_success_rate(suite_name)
            weighted_score += success_rate * weight
        
        return weighted_score
    
    def _calculate_quality_metrics(self) -> Dict[str, Any]:
        """Calculate quality metrics across all test suites."""
        comprehensive_rate = self._get_success_rate('comprehensive')
        fault_rate = self._get_success_rate('fault_tolerance')
        security_rate = self._get_success_rate('security')
        
        return {
            'reliability_score': (comprehensive_rate + fault_rate) / 2,
            'security_score': security_rate,
            'performance_validated': comprehensive_rate >= 90,
            'fault_tolerance_validated': fault_rate >= 85,
            'security_hardened': security_rate >= 95,
            'enterprise_ready': all([
                comprehensive_rate >= 90,
                fault_rate >= 85,
                security_rate >= 95
            ])
        }
    
    def _print_master_summary(self, report: Dict[str, Any]):
        """Print comprehensive master test summary."""
        print("=" * 80)
        print("🎯 NAMEL3SS PRODUCTION VALIDATION COMPLETE")
        print("=" * 80)
        
        # Overall statistics
        stats = report['aggregate_statistics']
        print(f"📊 OVERALL RESULTS:")
        print(f"   Total Tests: {stats['total_tests']}")
        print(f"   Passed: {stats['passed_tests']}")
        print(f"   Failed: {stats['failed_tests']}")
        print(f"   Success Rate: {stats['overall_success_rate']:.1f}%")
        print(f"   Total Duration: {stats['total_duration_seconds']:.1f} seconds")
        print()
        
        # Individual suite performance
        print(f"📈 SUITE PERFORMANCE:")
        for suite_name in ['comprehensive', 'fault_tolerance', 'security']:
            rate = self._get_success_rate(suite_name)
            status = "✅" if rate >= 90 else "⚠️" if rate >= 80 else "❌"
            print(f"   {status} {suite_name.replace('_', ' ').title()}: {rate:.1f}%")
        print()
        
        # Production readiness
        readiness = report['production_readiness']
        confidence = readiness['confidence_score']
        ready = readiness['ready_for_production']
        
        print(f"🚀 PRODUCTION READINESS:")
        print(f"   Status: {'✅ READY' if ready else '❌ NOT READY'}")
        print(f"   Confidence Score: {confidence:.1f}%")
        print()
        
        # Quality metrics
        quality = report['quality_metrics']
        print(f"🏆 QUALITY METRICS:")
        print(f"   Reliability Score: {quality['reliability_score']:.1f}%")
        print(f"   Security Score: {quality['security_score']:.1f}%")
        print(f"   Enterprise Ready: {'✅ YES' if quality['enterprise_ready'] else '❌ NO'}")
        print()
        
        # Deployment recommendations
        print(f"💡 DEPLOYMENT RECOMMENDATIONS:")
        for recommendation in report['deployment_recommendations']:
            print(f"   {recommendation}")
        print()
        
        print("=" * 80)
        
        if ready:
            print("🎉 CONGRATULATIONS! Namel3ss is ready for production deployment!")
        else:
            print("⚠️ Additional work required before production deployment.")
        
        print("=" * 80)


async def main():
    """Run the complete master test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Namel3ss Complete Test Suite')
    parser.add_argument('--quick', action='store_true', 
                      help='Run quick validation (skip some performance tests)')
    parser.add_argument('--output', default='master_test_report.json',
                      help='Output file for detailed report')
    
    args = parser.parse_args()
    
    # Run complete test suite
    runner = MasterTestRunner()
    report = await runner.run_complete_test_suite(quick_mode=args.quick)
    
    # Save comprehensive report
    with open(args.output, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n📄 Detailed master report saved to: {args.output}")
    
    # Return success if production ready
    return report['production_readiness']['ready_for_production']


if __name__ == "__main__":
    import sys
    
    result = asyncio.run(main())
    sys.exit(0 if result else 1)