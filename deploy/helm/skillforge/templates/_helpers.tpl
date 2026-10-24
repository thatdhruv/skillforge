{{- define "sf.name" -}}{{ .Release.Name }}{{- end -}}
{{- define "sf.labels" -}}
app.kubernetes.io/part-of: skillforge
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end -}}
{{- define "sf.img" -}}{{ .root.Values.image.registry }}{{ .name }}:{{ .root.Values.image.tag }}{{- end -}}
{{- define "sf.pgSecret" -}}{{ include "sf.name" . }}-postgres{{- end -}}
{{- define "sf.dbUrl" -}}
{{- if .Values.postgres.enabled -}}
postgres://{{ .Values.postgres.auth.user }}:$(POSTGRES_PASSWORD)@{{ include "sf.name" . }}-postgres:5432/{{ .Values.postgres.auth.database }}
{{- else -}}{{ .Values.postgres.externalUrl }}{{- end -}}
{{- end -}}
{{- define "sf.brokers" -}}
{{- if .Values.redpanda.enabled -}}{{ include "sf.name" . }}-redpanda:9092{{- else -}}{{ .Values.redpanda.externalBrokers }}{{- end -}}
{{- end -}}
{{- define "sf.podSecurity" -}}
automountServiceAccountToken: false
securityContext:
  runAsNonRoot: true
  seccompProfile: { type: RuntimeDefault }
{{- end -}}
{{- define "sf.containerSecurity" -}}
securityContext:
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  capabilities: { drop: ["ALL"] }
{{- end -}}
