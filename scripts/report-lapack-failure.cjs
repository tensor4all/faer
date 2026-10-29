// Called only by the scheduled/manual workflow; offline tests mock every API.
module.exports = async ({github, context, result, server, unexpected = '', stale = ''}) => {
  if (!['failure', 'cancelled'].includes(result)) return;
  const repo = context.repo;
  const label = 'scheduled-lapack-validation';
  const title = 'Scheduled LAPACK numerical validation is failing';
  try {
    await github.rest.issues.getLabel({...repo, name: label});
  } catch (error) {
    if (error.status !== 404) throw error;
    await github.rest.issues.createLabel({
      ...repo, name: label, color: 'B60205',
      description: 'Failures in the scheduled bounded LAPACK numerical validation',
    });
  }
  const issues = await github.paginate(github.rest.issues.listForRepo, {
    ...repo, state: 'open', labels: label, per_page: 100,
  });
  const issue = issues.find(item => !item.pull_request && item.title === title);
  const run = `${server}/${repo.owner}/${repo.repo}/actions/runs/${context.runId}`;
  const body = [
    `Latest failed run: ${run}`,
    `Commit: ${context.sha}`,
    `Validation result: ${result}`,
    '',
    'See the lapack-validation artifact: environment, build timing, exact test selection, per-family logs and summary.json.',
    '',
    unexpected ? `New failures (not listed in scripts/lapack-known-failures.json):\n\n${unexpected}` : '',
    stale ? `Listed as a known failure but passing now - remove it from scripts/lapack-known-failures.json:\n\n${stale}` : '',
    '',
    'Known failures are listed with their issue in scripts/lapack-known-failures.json and do not fire this issue.',
    'Coverage, omissions and local reproduction: docs/lapack-validation.md.',
  ].filter(line => line !== undefined && line !== null && line !== '').join('\n');
  if (issue) {
    await github.rest.issues.update({...repo, issue_number: issue.number, body});
  } else {
    await github.rest.issues.create({...repo, title, body, labels: [label]});
  }
};
