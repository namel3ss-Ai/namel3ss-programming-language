# CI/CD Quick Reference

Fast reference for the N3 Graph Editor CI/CD pipeline.

## 🚀 Quick Commands

### Deploy to Staging
```bash
git checkout develop
git push origin develop
# Automatic deployment triggered
```

### Deploy to Production
```bash
git tag v1.0.0
git push origin v1.0.0
# Manual approval required
```

### Monitor Deployment
```bash
kubectl get pods -n n3-production -w
kubectl logs -f deployment/backend -n n3-production
```

### Rollback
```bash
kubectl rollout undo deployment/backend -n n3-production
```

## 📊 Workflows

| Workflow | Trigger | Duration | Purpose |
|----------|---------|----------|---------|
| **Lint** | Push/PR | ~2 min | Code quality checks |
| **Unit Tests** | Push/PR | ~5 min | Python + Frontend tests |
| **E2E Tests** | Push/PR | ~6 min | Browser automation tests |
| **Build** | Push/Tags | ~10 min | Docker image builds |
| **Deploy Staging** | Push to `develop` | ~3 min | Staging deployment |
| **Deploy Production** | Tags `v*` | ~5 min | Production deployment |

**Total Pipeline**: ~30 minutes (parallel execution)

## 🌍 Environments

### Staging
- **URL**: https://staging.ai-graph-editor.com
- **Branch**: `develop`
- **Auto-deploy**: Yes
- **Database**: 10Gi
- **Replicas**: 2-10 (auto-scale)

### Production
- **URL**: https://n3-graph-editor.com
- **Tags**: `v*` (semantic versioning)
- **Auto-deploy**: Yes (with approval)
- **Database**: 50Gi (StatefulSet)
- **Replicas**: 3-20 (auto-scale)
- **High Availability**: Yes
- **Auto-rollback**: Yes

## 🐳 Docker Images

```bash
# Backend
ghcr.io/ssebowadisan/namel3ss-programming-language-backend:latest
ghcr.io/ssebowadisan/namel3ss-programming-language-backend:v1.0.0

# Frontend
ghcr.io/ssebowadisan/namel3ss-programming-language-frontend:latest
ghcr.io/ssebowadisan/namel3ss-programming-language-frontend:v1.0.0

# YJS Server
ghcr.io/ssebowadisan/namel3ss-programming-language-yjs-server:latest
ghcr.io/ssebowadisan/namel3ss-programming-language-yjs-server:v1.0.0
```

## 🔑 Required Secrets

### GitHub Repository Secrets
```
KUBE_CONFIG_STAGING
KUBE_CONFIG_PRODUCTION
STAGING_DATABASE_URL
PRODUCTION_DATABASE_URL
STAGING_SECRET_KEY
PRODUCTION_SECRET_KEY
SLACK_WEBHOOK
SMOKE_TEST_USER
SMOKE_TEST_PASSWORD
CODECOV_TOKEN
```

## 📝 Deployment Checklist

### Before Deploying
- [ ] All tests passing
- [ ] Code reviewed and approved
- [ ] Changelog updated
- [ ] Database migrations prepared
- [ ] Secrets configured
- [ ] Staging tested

### Deployment
- [ ] Tag version (semantic: `vX.Y.Z`)
- [ ] Push tag to trigger pipeline
- [ ] Monitor rollout status
- [ ] Run smoke tests
- [ ] Check application logs
- [ ] Verify metrics

### After Deployment
- [ ] Create GitHub Release
- [ ] Update documentation
- [ ] Notify team (Slack)
- [ ] Monitor for errors
- [ ] Backup database

## 🔍 Monitoring

### Health Checks
```bash
# Backend
curl https://n3-graph-editor.com/api/health

# Frontend
curl https://n3-graph-editor.com/

# Full smoke test
curl -X POST https://n3-graph-editor.com/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"test","password":"test"}'
```

### Kubernetes Status
```bash
# Pods
kubectl get pods -n n3-production

# Deployments
kubectl get deployments -n n3-production

# Services
kubectl get services -n n3-production

# Ingress
kubectl get ingress -n n3-production

# HPA (Auto-scaling)
kubectl get hpa -n n3-production
```

### Logs
```bash
# Backend logs
kubectl logs -f deployment/backend -n n3-production

# Frontend logs
kubectl logs -f deployment/frontend -n n3-production

# YJS Server logs
kubectl logs -f deployment/yjs-server -n n3-production

# All pods
kubectl logs -f -l app=backend -n n3-production
```

## 🚨 Troubleshooting

### Pod Not Starting
```bash
kubectl describe pod <pod-name> -n n3-production
kubectl logs <pod-name> -n n3-production
```

### Image Pull Error
```bash
# Recreate registry secret
kubectl delete secret ghcr-secret -n n3-production
# Redeploy workflow
```

### Database Connection Failed
```bash
# Check database pod
kubectl logs deployment/postgres -n n3-production

# Test connection
kubectl exec -it deployment/backend -n n3-production -- \
  python -c "from n3_server.db.session import engine; print(engine)"
```

### Rollback Deployment
```bash
# Automatic rollback (on failure)
# Manual rollback
kubectl rollout undo deployment/backend -n n3-production
kubectl rollout undo deployment/frontend -n n3-production
kubectl rollout undo deployment/yjs-server -n n3-production
```

### Check Rollout History
```bash
kubectl rollout history deployment/backend -n n3-production
kubectl rollout history deployment/frontend -n n3-production
```

### Scale Manually
```bash
kubectl scale deployment/backend --replicas=5 -n n3-production
kubectl scale deployment/frontend --replicas=5 -n n3-production
```

## 🔧 Manual Operations

### Update ConfigMap
```bash
kubectl edit configmap n3-config -n n3-production
kubectl rollout restart deployment/backend -n n3-production
```

### Update Secret
```bash
kubectl create secret generic n3-secrets \
  --from-literal=DATABASE_URL="..." \
  --from-literal=SECRET_KEY="..." \
  --dry-run=client -o yaml | kubectl apply -n n3-production -f -
kubectl rollout restart deployment/backend -n n3-production
```

### Database Backup
```bash
kubectl exec -n n3-production deployment/postgres -- \
  pg_dump -U n3 n3_graphs > backup-$(date +%Y%m%d).sql
```

### Database Restore
```bash
cat backup-20250120.sql | kubectl exec -i -n n3-production deployment/postgres -- \
  psql -U n3 n3_graphs
```

## 📈 Performance

### Resource Usage
```bash
# Nodes
kubectl top nodes

# Pods
kubectl top pods -n n3-production

# Specific pod
kubectl top pod <pod-name> -n n3-production
```

### Auto-scaling Status
```bash
# View HPA
kubectl get hpa -n n3-production

# Describe HPA
kubectl describe hpa backend-hpa -n n3-production

# Current metrics
kubectl get hpa backend-hpa -n n3-production -o yaml
```

## 🎯 Common Tasks

### Create New Release
```bash
# 1. Update version in code
# 2. Commit changes
git add .
git commit -m "Bump version to 1.1.0"
git push origin main

# 3. Create tag
git tag -a v1.1.0 -m "Release v1.1.0"
git push origin v1.1.0

# 4. Monitor deployment
kubectl get pods -n n3-production -w
```

### Hotfix Deployment
```bash
# 1. Create hotfix branch
git checkout -b hotfix/urgent-fix main

# 2. Make fix
# 3. Test locally

# 4. Merge to main
git checkout main
git merge hotfix/urgent-fix
git push origin main

# 5. Create patch version tag
git tag v1.0.1
git push origin v1.0.1
```

### View Deployment Status
```bash
# All deployments
kubectl get deployments -n n3-production

# Specific deployment
kubectl rollout status deployment/backend -n n3-production

# Deployment history
kubectl rollout history deployment/backend -n n3-production
```

## 🔗 Useful Links

- [GitHub Actions](https://github.com/SsebowaDisan/namel3ss-programming-language/actions)
- [Container Registry](https://github.com/SsebowaDisan/namel3ss-programming-language/pkgs/container/namel3ss-programming-language-backend)
- [Releases](https://github.com/SsebowaDisan/namel3ss-programming-language/releases)
- [Full Documentation](./CICD_DEPLOYMENT.md)

## 💡 Tips

1. **Always test in staging first**
   ```bash
   git push origin develop
   # Wait for staging deployment
   # Test at https://staging.ai-graph-editor.com
   ```

2. **Use semantic versioning**
   - `v1.0.0` - Major release
   - `v1.1.0` - Minor features
   - `v1.0.1` - Patches

3. **Monitor after deployment**
   ```bash
   watch -n 2 'kubectl get pods -n n3-production'
   ```

4. **Check logs immediately**
   ```bash
   kubectl logs -f --tail=100 deployment/backend -n n3-production
   ```

5. **Verify health endpoint**
   ```bash
   curl https://n3-graph-editor.com/api/health
   ```

---

**Quick Tip**: Use `kubectl port-forward` for local debugging:
```bash
kubectl port-forward deployment/backend 8000:8000 -n n3-production
curl http://localhost:8000/health
```
