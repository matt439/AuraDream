// AuraDream's page script: the automatic startup backup settings in the
// Bundles dialog (protection, consent, retention cleanup and credential
// status, read only while the dialog is open). Declarations only;
// bundles.js's initBundles calls initAutomaticBackup.
'use strict';

// Automatic backup settings are read only while Bundles is open. Keep the
// edited draft and its revision apart from polls, including another tab's saves.
let automaticBackupStatus = null;
let automaticBackupDraftRevision = null;
let automaticBackupDraftDestination = null;
let automaticBackupDraftModeRevision = null;
let automaticBackupDraftCredentialGeneration = null;
let automaticBackupDirty = false;
let automaticBackupEditEpoch = 0;
let automaticBackupReading = false;
let automaticBackupSaving = false;
let automaticBackupUnavailable = false;
let automaticBackupEpoch = 0;

function automaticBackupElements() {
  return {
    enabled: document.getElementById('automatic-backup-enabled'),
    age: document.getElementById('automatic-backup-age'),
    keep: document.getElementById('automatic-backup-keep'),
    consent: document.getElementById('automatic-backup-consent'),
    protection: document.getElementById('automatic-backup-protection'),
    encryptedConsent: document.getElementById('automatic-backup-encrypted-consent'),
    recoveryConsent: document.getElementById('automatic-backup-recovery-consent'),
    credentialStatus: document.getElementById('automatic-backup-credential-status'),
    retentionEnabled: document.getElementById('automatic-backup-retention-enabled'),
    deleteConsent: document.getElementById('automatic-backup-delete-consent'),
    save: document.getElementById('automatic-backup-save'),
    reload: document.getElementById('automatic-backup-reload'),
    status: document.getElementById('automatic-backup-status'),
    warning: document.getElementById('automatic-backup-warning'),
    destination: document.getElementById('automatic-backup-destination'),
    mode: document.getElementById('automatic-backup-mode'),
    retentionStatus: document.getElementById('automatic-backup-retention-status'),
    retentionWarning: document.getElementById('automatic-backup-retention-warning'),
    feedback: document.getElementById('automatic-backup-feedback'),
  };
}

function automaticBackupSay(message, failed = false) {
  const feedback = automaticBackupElements().feedback;
  if (feedback.textContent !== message) feedback.textContent = message;
  feedback.classList.toggle('failed', failed);
}

function automaticBackupOutcomeText(status) {
  const reasons = {
    disabled: 'Automatic startup backups are off.',
    next_start: 'Automatic backup is enabled for the next server start. No backup is created by saving settings.',
    legacy_settings: 'Earlier automatic backup settings need explicit confirmation before they can run.',
    destination_changed: 'The backup destination changed. Confirm the new destination before automatic backups can run.',
    settings_unreadable: 'Automatic backups are off because their settings cannot be read. Repair the settings file before saving.',
    destination_unavailable: 'Automatic backups are off because the backup destination is unavailable.',
    not_started: 'This server has not attempted an automatic backup.',
    checking: 'Checking whether the automatic startup backup is due. The gallery is kept.',
    empty: 'Automatic startup backup skipped: the gallery is empty. Earlier copies are kept.',
    young: 'Automatic startup backup is not due: a verified automatic copy is recent enough.',
    clock_backwards: 'Automatic startup backup skipped because the clock moved backward. Earlier copies are kept.',
    admission_busy: 'Automatic startup backup deferred because other work is active. There is no automatic retry this session.',
    gallery_busy: 'Automatic startup backup deferred because another process is using the gallery. There is no automatic retry this session.',
    untrusted_inventory: 'Automatic startup backup skipped because archive ownership could not be confirmed.',
    operation_already_recorded: 'This automatic startup backup attempt was already recorded; no duplicate was created.',
    verified_publication: 'Automatic startup backup saved and verified. The gallery is unchanged.',
    receipt_not_saved: 'The automatic backup file was saved, but its ownership receipt could not be saved. It remains unmanaged; earlier copies are kept.',
    journal_unusable: 'Automatic startup backup failed because its receipt journal needs repair. Earlier copies are kept.',
    source_mismatch: 'Automatic startup backup failed because the gallery identity did not match. Earlier copies are kept.',
    destination_mismatch: 'Automatic startup backup failed because the destination did not match. Earlier copies are kept.',
    destination_identity_mismatch: 'Automatic startup backup failed because its saved destination identity did not match. Earlier copies are kept.',
    invalid_policy: 'Automatic startup backup failed because its saved policy is invalid. Earlier copies are kept.',
    invalid_identity_or_protection: 'Automatic startup backup failed because its identity or protection settings are invalid. Earlier copies are kept.',
    credential_missing: 'The configured encrypted-backup credential is missing. Inspect the local profile and repair its offline setup using the intended Windows account. No plaintext fallback was used.',
    credential_session_unavailable: 'This Windows session cannot read the configured backup credential. Use the intended Windows account and an available sign-in session before the next start. No plaintext fallback was used.',
    credential_access_denied: 'Windows refused access to the configured backup credential. Review credential-store access for the intended account before the next start. No plaintext fallback was used.',
    credential_unusable: 'The configured backup credential could not be used. Inspect the local profile and repair the offline credential setup before the next start. No plaintext fallback was used.',
    worker_cancelled_before_start: 'Automatic startup backup stopped before its worker started. No new backup was created.',
    worker_start_failed: 'Automatic startup backup could not start its worker. Earlier copies are kept.',
    backup_failed: 'Automatic startup backup failed. Earlier copies are kept; check the destination and settings before the next server start.',
  };
  return reasons[status.reason] || ({
    disabled: 'Automatic startup backups are off.',
    needs_confirmation: 'Automatic backup settings need explicit confirmation before they can run.',
    unavailable: 'Automatic startup backups are unavailable; review the warning below.',
    scheduled: 'Automatic backup is enabled for the next server start.',
    running: 'Automatic startup backup is running. The gallery is kept.',
    succeeded: 'Automatic startup backup saved and verified. The gallery is unchanged.',
    failed: 'Automatic startup backup failed. Earlier copies are kept.',
    deferred: 'Automatic startup backup was deferred. There is no automatic retry this session.',
    skipped: 'Automatic startup backup was skipped. Earlier copies are kept.',
    unmanaged: 'The saved automatic backup has no confirmed receipt. Earlier copies are kept.',
  }[status.state] || 'This server has not attempted an automatic backup.');
}

function automaticBackupRetentionText(status) {
  if (status.retention_pending) {
    return 'Automatic cleanup is blocked by an unresolved deletion attempt. Physical removal is unconfirmed; do not assume that copy was preserved. Repair the pending state before cleanup can run again. There is no automatic retry.';
  }
  if (status.retention_uncertain) {
    return 'The cleanup journal could not be inspected. Whether an earlier removal completed is unknown. Repair the journal before enabling cleanup again. There is no automatic retry.';
  }
  const marked = Number.isInteger(status.retention_marked) ? status.retention_marked : 0;
  const retained = Number.isInteger(status.retention_retained) ? status.retention_retained : 0;
  if (marked > 0) {
    return `${marked} older automatic ${marked === 1 ? 'copy was' : 'copies were'} marked for removal. Physical removal is unconfirmed.`
      + (retained > 0 ? ` ${retained} older ${retained === 1 ? 'candidate was' : 'candidates were'} not marked.` : '');
  }
  if (retained > 0) {
    return `No older automatic copies were marked for removal. ${retained} older ${retained === 1 ? 'candidate was' : 'candidates were'} not marked.`;
  }
  if (status.retention_state === 'succeeded' && status.retention_reason === 'completed') return 'Cleanup finished without marking an older automatic copy.';
  if (status.retention_state === 'skipped') return 'Automatic cleanup was skipped; no older copy was marked for removal this attempt. There is no automatic retry this session.';
  if (status.retention_warning || ['blocked', 'failed', 'unavailable'].includes(status.retention_state)) {
    return 'Automatic cleanup is unavailable or incomplete; review the cleanup warning. There is no automatic retry this session.';
  }
  if (status.retention_enabled) {
    return 'Cleanup is enabled after a new verified automatic startup backup. The new copy and newest confirmed copies are kept; manual, foreign, uncertain and busy copies are protected. Saving or reloading settings creates no backup and deletes no file.';
  }
  return 'Automatic cleanup is not enabled for future backups. The requested count is not enforced.';
}

function automaticBackupCredentialText(status) {
  if (status.credential_state === 'pending') {
    return 'Offline credential setup is pending. Startup backups cannot be enabled until the setup outcome is resolved. Use the offline inspect command for local metadata; this page never checks the credential store.';
  }
  if (status.credential_configured && status.credential_state === 'ready') {
    return `An encrypted credential reference is configured${status.credential_generation ? ` for generation ${status.credential_generation}` : ''}. Its availability has not been checked here. The backup worker reads it only when an encrypted backup is due; saving, reloading and status do not check the credential store.`;
  }
  return 'Encrypted startup backups need an offline credential setup for this gallery and destination. This page accepts no password and never reuses the manual bundle password fields.';
}

function automaticBackupEncryptionIssue() {
  const status = automaticBackupStatus;
  if (status?.credential_state === 'pending') return 'Resolve the pending offline credential setup before enabling encrypted backups';
  if (!status?.credential_configured || status.credential_state !== 'ready'
      || status.credential_provider !== 'windows-credential-manager-v1' || !status.credential_generation
      || status.protection !== 'encrypted' || status.reason === 'destination_changed') {
    return 'Complete a new offline encrypted credential setup for this gallery and destination, then reload the settings';
  }
  if (automaticBackupDraftCredentialGeneration !== status.credential_generation) {
    return 'Reload saved settings and confirm the changed credential generation before enabling encrypted backups';
  }
  return '';
}

function automaticBackupRetentionIssue() {
  if (!automaticBackupStatus || automaticBackupUnavailable) return 'Read the automatic backup settings before enabling cleanup';
  if (automaticBackupStatus.retention_pending) return 'Repair the unresolved deletion attempt before enabling cleanup';
  if (automaticBackupStatus.retention_uncertain) return 'Repair the unreadable cleanup journal before enabling cleanup';
  if (!automaticBackupStatus.mode_revision || !automaticBackupStatus.shared_mode_id) {
    return 'Install Windows shared archive mode offline before enabling cleanup, then reload these settings';
  }
  return '';
}

function receiveAutomaticBackup(status, replaceDraft = false) {
  automaticBackupStatus = status;
  automaticBackupUnavailable = false;
  const fields = automaticBackupElements();
  const stamp = status.verified_at ? ` Verified at ${new Date(status.verified_at).toLocaleString()}.` : '';
  const text = automaticBackupOutcomeText(status) + stamp;
  if (fields.status.textContent !== text) fields.status.textContent = text;
  fields.warning.textContent = status.warning || '';
  fields.warning.hidden = !status.warning;
  fields.retentionWarning.textContent = status.retention_warning || '';
  fields.retentionWarning.hidden = !status.retention_warning;
  fields.retentionStatus.textContent = automaticBackupRetentionText(status);
  fields.credentialStatus.textContent = automaticBackupCredentialText(status);
  fields.mode.textContent = status.shared_mode_id && status.mode_revision
    ? `Installed shared archive mode: ${status.shared_mode_id}. Cleanup permission applies to this mode and destination.`
    : 'Cleanup needs Windows shared archive mode installed offline while every server using this destination is stopped. Use the documented offline shared-mode installer, then Reload saved settings. This page never installs it.';
  fields.destination.textContent = status.destination
    ? `Automatic backup destination: ${status.destination}` : 'The automatic backup destination is unavailable.';
  if (replaceDraft || !automaticBackupDirty) {
    fields.enabled.checked = !!status.enabled;
    fields.age.value = status.minimum_age_hours ?? '';
    fields.keep.value = status.keep_last ?? '';
    fields.consent.checked = false;
    fields.protection.value = status.protection || '';
    fields.encryptedConsent.checked = false;
    fields.recoveryConsent.checked = false;
    fields.retentionEnabled.checked = !!status.retention_enabled;
    fields.deleteConsent.checked = false;
    automaticBackupDraftRevision = status.revision;
    automaticBackupDraftDestination = status.destination;
    automaticBackupDraftModeRevision = status.mode_revision || null;
    automaticBackupDraftCredentialGeneration = status.credential_generation || null;
    automaticBackupDirty = false;
  }
  syncAutomaticBackupControls();
}

async function refreshAutomaticBackup(replaceDraft = false) {
  if (!els.bundlesDialog.open || automaticBackupReading || automaticBackupSaving) return;
  const epoch = automaticBackupEpoch;
  const draftEpoch = automaticBackupEditEpoch;
  automaticBackupReading = true;
  syncAutomaticBackupControls();
  try {
    const status = await api('/api/gallery/automatic-backup' + (replaceDraft ? '/reload' : ''),
      replaceDraft ? { method: 'POST' } : undefined);
    if (epoch === automaticBackupEpoch) {
      const newerDraft = draftEpoch !== automaticBackupEditEpoch;
      receiveAutomaticBackup(status, replaceDraft && !newerDraft);
      if (replaceDraft) automaticBackupSay(newerDraft
        ? 'Saved settings were read. Your newer unsaved edits are kept.' : 'Saved settings reloaded.');
    }
  } catch (err) {
    if (epoch === automaticBackupEpoch) {
      automaticBackupUnavailable = true;
      automaticBackupSay(`Automatic backup settings could not be read. ${err.message}`, true);
    }
  } finally {
    automaticBackupReading = false;
    syncAutomaticBackupControls();
  }
}

function automaticBackupIssue() {
  const fields = automaticBackupElements();
  const held = automaticBackupSaving ? 'Wait for these settings to finish saving'
    : automaticBackupReading ? 'Wait for the automatic backup settings'
      : !automaticBackupStatus || automaticBackupUnavailable ? 'Read the automatic backup settings first'
        : automaticBackupStatus.revision === null ? 'Repair the unreadable automatic backup settings file first'
          : bundleActionHold();
  if (held) return { reason: held };
  if (!automaticBackupDirty) return { reason: 'Change a setting before saving' };
  if (!fields.enabled.checked) return { reason: '' };
  for (const [input, label] of [[fields.age, 'Minimum age'], [fields.keep, 'Requested copies']]) {
    const value = input.valueAsNumber;
    if (input.validity.badInput || !Number.isInteger(value) || value < 1 || value > Number(input.max)) {
      return { reason: `${label} must be a whole number from 1 to ${input.max}`, field: input };
    }
  }
  if (!automaticBackupStatus.destination) return { reason: 'Read a valid backup destination before enabling automatic backups' };
  if (automaticBackupDraftDestination !== automaticBackupStatus.destination) {
    return { reason: 'Reload saved settings and confirm the changed backup destination before enabling automatic backups' };
  }
  if (!['unencrypted', 'encrypted'].includes(fields.protection.value)) {
    return { reason: 'Choose archive protection before enabling automatic backups', field: fields.protection };
  }
  if (automaticBackupStatus.credential_state === 'pending') {
    return { reason: 'Resolve the pending offline credential setup before enabling automatic backups', field: fields.protection };
  }
  if (fields.protection.value === 'encrypted') {
    const encryptionIssue = automaticBackupEncryptionIssue();
    if (encryptionIssue) return { reason: encryptionIssue, field: fields.protection };
    if (!fields.encryptedConsent.checked) return { reason: 'Explicitly allow unattended encrypted archives with the configured credential reference', field: fields.encryptedConsent };
    if (!fields.recoveryConsent.checked) return { reason: 'Confirm independent password recovery outside this Windows account and PC', field: fields.recoveryConsent };
  } else if (!fields.consent.checked) {
    return { reason: 'Explicitly allow unencrypted images and prompts at the displayed destination', field: fields.consent };
  }
  if (fields.retentionEnabled.checked) {
    const retentionIssue = automaticBackupRetentionIssue();
    if (retentionIssue) return { reason: retentionIssue, field: fields.retentionEnabled };
    if (automaticBackupDraftModeRevision !== automaticBackupStatus.mode_revision) {
      return { reason: 'Reload saved settings and confirm the changed shared archive mode before enabling cleanup', field: fields.retentionEnabled };
    }
    if (!fields.deleteConsent.checked) {
      return { reason: 'Explicitly allow irreversible deletion at the displayed destination with the chosen count', field: fields.deleteConsent };
    }
  }
  return { reason: '' };
}

function syncAutomaticBackupControls() {
  const fields = automaticBackupElements();
  if (!fields.save) return;
  const issue = automaticBackupIssue();
  fields.save.setAttribute('aria-disabled', String(!!issue.reason));
  fields.save.title = issue.reason || 'Save these choices for the next server start; no backup is created now';
  const held = automaticBackupReading || automaticBackupSaving;
  fields.reload.setAttribute('aria-disabled', String(held));
  fields.reload.title = held ? 'Wait for the current settings request' : 'Discard unsaved edits and read the saved settings';
  const retentionIssue = automaticBackupRetentionIssue();
  fields.retentionEnabled.setAttribute('aria-disabled', String(!fields.retentionEnabled.checked && !!retentionIssue));
  fields.retentionEnabled.title = retentionIssue || 'Permit cleanup only after a new verified automatic backup; saving deletes no file';
}

async function saveAutomaticBackup() {
  const fields = automaticBackupElements();
  const issue = automaticBackupIssue();
  if (issue.reason) {
    automaticBackupSay(issue.reason + '.', true);
    if (issue.field) focusBundleField(issue.field);
    return;
  }
  const body = { revision: automaticBackupDraftRevision, enabled: fields.enabled.checked };
  if (body.enabled) Object.assign(body, {
    minimum_age_hours: fields.age.valueAsNumber, keep_last: fields.keep.valueAsNumber,
    protection: fields.protection.value, destination: automaticBackupDraftDestination,
    retention_enabled: fields.retentionEnabled.checked,
    allow_managed_deletion: fields.retentionEnabled.checked && fields.deleteConsent.checked,
    expected_mode_revision: fields.retentionEnabled.checked ? automaticBackupDraftModeRevision : null,
  });
  if (body.enabled && body.protection === 'encrypted') Object.assign(body, {
    allow_encrypted: fields.encryptedConsent.checked, confirm_recovery: fields.recoveryConsent.checked,
  });
  else if (body.enabled) body.allow_unencrypted = fields.consent.checked;
  const draftEpoch = automaticBackupEditEpoch;
  ++automaticBackupEpoch;
  automaticBackupSaving = true;
  fields.deleteConsent.checked = false;
  fields.encryptedConsent.checked = false;
  fields.recoveryConsent.checked = false;
  syncAutomaticBackupControls();
  automaticBackupSay('Saving settings for the next server start. No backup is created and no file is deleted now.');
  try {
    const status = await api('/api/gallery/automatic-backup', {
      method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const newerDraft = draftEpoch !== automaticBackupEditEpoch;
    receiveAutomaticBackup(status, !newerDraft);
    // This response acknowledges our own saved base. Keep edits made while
    // waiting, but let their next save build on that base without a false
    // conflict. A status poll or another tab's save never advances this token.
    if (newerDraft) automaticBackupDraftRevision = status.revision;
    automaticBackupSay((body.enabled
      ? 'Saved for the next server start. No backup was created and no file was deleted now. '
        + (body.retention_enabled ? 'Cleanup permission applies only after a new verified automatic backup.' : 'Automatic cleanup is not enabled for future backups; the requested count is not enforced.')
      : 'Automatic backups and cleanup are off for the next server start. Saving these settings deleted no file.')
      + (newerDraft ? ' Your newer unsaved edits are kept.' : ''));
  } catch (err) {
    automaticBackupSay(`${err.message} Your unsaved edits are kept; reload saved settings before trying again if another tab changed them.`, true);
  } finally {
    automaticBackupSaving = false;
    fields.deleteConsent.checked = false;
    fields.encryptedConsent.checked = false;
    fields.recoveryConsent.checked = false;
    syncAutomaticBackupControls();
  }
}

function initAutomaticBackup() {
  const fields = automaticBackupElements();
  for (const input of [fields.enabled, fields.age, fields.keep, fields.consent, fields.protection, fields.encryptedConsent, fields.recoveryConsent, fields.retentionEnabled, fields.deleteConsent]) input.addEventListener('input', () => {
    if (input === fields.protection) fields.consent.checked = false;
    if (input !== fields.deleteConsent) fields.deleteConsent.checked = false;
    if (![fields.encryptedConsent, fields.recoveryConsent, fields.deleteConsent].includes(input)) {
      fields.encryptedConsent.checked = false;
      fields.recoveryConsent.checked = false;
    }
    ++automaticBackupEditEpoch;
    automaticBackupDirty = true;
    syncAutomaticBackupControls();
  });
  fields.retentionEnabled.addEventListener('click', event => {
    const issue = automaticBackupRetentionIssue();
    if (fields.retentionEnabled.checked && issue) {
      event.preventDefault();
      automaticBackupSay(issue + '.', true);
    }
  });
  fields.save.onclick = () => { void saveAutomaticBackup(); };
  fields.reload.onclick = () => {
    if (fields.reload.getAttribute('aria-disabled') === 'true') {
      automaticBackupSay(fields.reload.title + '.', true);
      return;
    }
    void refreshAutomaticBackup(true);
  };
}
