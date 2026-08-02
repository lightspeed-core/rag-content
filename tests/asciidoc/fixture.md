# Getting Started with Lightspeed

# Prerequisites

Before installing, ensure you have cluster administrator access
to your OpenShift environment.

The following tools must be available:

* oc command-line client
* podman for building container images
* Access to a container registry

# Installation

Enable the feature by running the following command:

```shell
oc apply -f config.yaml
```

Verify the installation succeeded:

1. Check that the pod is running.
2. Confirm the service endpoint responds.
3. Review the logs for errors.

[NOTE]
----
The command requires admin privileges.
----

# Configuration

The main configuration file is config.yaml.

proxy_url:: The URL of the upstream proxy.
timeout:: Connection timeout in seconds.
