# ☄️ GPUs Elasticity App (WIP)
Maximize your GPUs Kubernetes resource utilization with flexible and efficient elastic quotas.

If you like this App, give us a star to show your support for the project ⭐

## 📚 Description
The GPUs Elasticity App extends the Kubernetes Resource Quotas for GPUs adding more flexibility through 2 custom resources: ElasticQuotas and CompositeElasticQuotas.

While standard Kubernetes Resource Quotas allows users to define limits on the maximum overall resource allocation of each namespace, our Elastic Quotas let users define 2 different limits: a minimum amount of guaranteed resources for the namespace and a maximum amount of resources that the namespace can consume. This allows namespaces to borrow reserved resource quotas from other namespaces that are not using them, as long as they do not exceed their maximum limit (if any) and the namespaces lending the quotas do not need them. When a namespace claims back its reserved min resources, pods borrowing resources from other namespaces are preempted to make up space.

GPUs Elasticity also differs from the standard Kubernetes quota management by computing the used quotas based on running Pods only, in order to avoid lower resource utilization due to scheduled Pods that failed to start.

The App extends the basic implementation with features such as over-quota pods preemption, CompositeElasticQuota resources, custom resource, fair sharing of over-quota resources, and optional maximum limits. CompositeElasticQuota are quotas associated with groups of namespaces, allowing users to set quotas at organization levels as opposed to team level.

Overall, GPUs Elasticity allows user to benefit from a more flexible and efficient resource quota management in your Kubernetes cluster. Try it out today, and reach out if you have any feedback!
