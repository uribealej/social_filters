// Excel timing export. Reads the generator's JSON records from stdin.
import fs from 'node:fs/promises';
import path from 'node:path';
import os from 'node:os';
import { createRequire } from 'node:module';
import { pathToFileURL } from 'node:url';

const localRequire = createRequire(import.meta.url);
let artifactEntry;
try {
  artifactEntry = localRequire.resolve('@oai/artifact-tool');
} catch {
  const runtime = path.join(os.homedir(), '.cache/codex-runtimes/codex-primary-runtime/dependencies/node');
  artifactEntry = createRequire(path.join(runtime, 'package.json')).resolve('@oai/artifact-tool');
}
const { Workbook, SpreadsheetFile } = await import(pathToFileURL(artifactEntry).href);
const chunks = [];
for await (const chunk of process.stdin) chunks.push(chunk);
const { records, generated_at: generatedAt } = JSON.parse(Buffer.concat(chunks).toString('utf8'));
if (!records?.length) throw new Error('No timing records to export.');
const outputPath = path.resolve(process.argv[2]);
const previewDir = process.argv[3];
const workbook = Workbook.create();
const timing = workbook.worksheets.add('Timing');
const frames = workbook.worksheets.add('Frames');
const first = 14;
const last = first + records.length - 1;
const navy = '#253B53';
const pale = '#EAF0F6';

function styleSheet(sheet, range) {
  sheet.showGridLines = false;
  sheet.getRange(range).format.font = { name: 'Arial', size: 10, color: '#202C39' };
  sheet.getRange(range).format.rowHeight = 22;
  sheet.getRange(range).format.verticalAlignment = 'center';
}
function header(sheet, range) {
  sheet.getRange(range).format = {
    fill: navy, font: { name: 'Arial', size: 10, bold: true, color: '#FFFFFF' },
    horizontalAlignment: 'center', verticalAlignment: 'center', wrapText: true,
  };
  sheet.getRange(range).format.rowHeight = 36;
}

styleSheet(timing, `A2:O${last + 5}`);
timing.tabColor = navy;
timing.getRange('A2').values = [['Experiment 9 timing']];
timing.getRange('A2').format.font = { name: 'Arial', size: 16, bold: true, color: navy };
timing.getRange('A4:E4').values = [['Hemifield', 'Conditions', 'CSV time (s)', 'Pauses (s)', 'Total time (s)']];
header(timing, 'A4:E4');
const sides = [...new Set(records.map(r => r.hemifield))];
for (let i = 0; i < sides.length; i++) {
  const row = 5 + i;
  timing.getRange(`A${row}`).values = [[sides[i]]];
  timing.getRange(`B${row}:E${row}`).formulas = [[
    `=COUNTIFS($B$${first}:$B$${last},A${row})`,
    `=SUMIFS($K$${first}:$K$${last},$B$${first}:$B$${last},A${row})`,
    `=SUMIFS($L$${first}:$L$${last},$B$${first}:$B$${last},A${row})+SUMIFS($M$${first}:$M$${last},$B$${first}:$B$${last},A${row})`,
    `=SUMIFS($N$${first}:$N$${last},$B$${first}:$B$${last},A${row})`,
  ]];
}
const totalRow = 5 + sides.length;
timing.getRange(`A${totalRow}`).values = [['All stimuli']];
timing.getRange(`B${totalRow}:E${totalRow}`).formulas = [[
  `=SUM(B5:B${totalRow - 1})`, `=SUM(C5:C${totalRow - 1})`,
  `=SUM(D5:D${totalRow - 1})`, `=SUM(E5:E${totalRow - 1})`,
]];
timing.getRange(`A${totalRow}:E${totalRow}`).format.fill = pale;
timing.getRange(`A${totalRow}:E${totalRow}`).format.font.bold = true;
timing.getRange(`C5:E${totalRow}`).setNumberFormat('0.000');
timing.getRange('A9').values = [['One presentation per condition. No initial static display. External pauses are excluded from CSV time.']];
timing.getRange('A10').values = [['The blank delay starts at the final visible flicker offset and ends at the first bout ON frame.']];
timing.getRange('A12').values = [['Source: parameters/stimulus_time_summary.csv. Generated ' + generatedAt]];
timing.getRange('A9:A12').format.font = { name: 'Arial', size: 10, italic: true, color: '#526273' };
timing.getRange('A13:O13').values = [[
  'Stimulus', 'Hemifield', 'Condition', 'Preconditioning', 'Pre target (s)', 'Pre actual (s)',
  'Delay target (s)', 'Delay actual (s)', 'Bout onset (s)', 'Bout time (s)', 'CSV time (s)',
  'Pause before (s)', 'Pause after (s)', 'Total time (s)', 'Pre ON pulses',
]];
header(timing, 'A13:O13');
timing.getRange(`A${first}:O${last}`).values = records.map(r => [
  r.stimulus_key, r.hemifield, r.condition, r.preconditioning,
  r.preconditioning_requested_sec, r.preconditioning_sec, r.delay_requested_sec, r.delay_sec,
  null, r.bout_sec, null, r.pause_before_sec, r.pause_after_sec, null, r.preconditioning_pulses,
]);
for (let i = 0; i < records.length; i++) {
  const row = first + i;
  timing.getRange(`I${row}`).formulas = [[`=SUM(F${row},H${row})`]];
  timing.getRange(`K${row}`).formulas = [[`=SUM(F${row},H${row},J${row})`]];
  timing.getRange(`N${row}`).formulas = [[`=SUM(K${row}:M${row})`]];
  if (i % 2) timing.getRange(`A${row}:O${row}`).format.fill = '#F4F6F9';
}
timing.getRange(`E${first}:N${last}`).setNumberFormat('0.000');
timing.getRange(`O${first}:O${last}`).setNumberFormat('0');
timing.getRange(`N${first}:N${last}`).format.font.bold = true;
timing.getRange(`N${first}:N${last}`).format.fill = '#E8F0E8';
timing.getRange(`A${last + 3}`).values = [['Timing is a generated record. Change the JSON configs and rerun the generator to update the stimuli and this workbook.']];
timing.getRange(`A${last + 4}`).values = [['Pre target / actual compare requested duration with complete ON pulses. All timing columns use seconds.']];
timing.getRange(`A${last + 3}:A${last + 4}`).format.font = { name: 'Arial', size: 10, italic: true, color: '#526273' };
timing.getRange(`A2:A${last}`).format.columnWidth = 23;
timing.getRange(`B2:B${last}`).format.columnWidth = 13;
timing.getRange(`C2:C${last}`).format.columnWidth = 16;
timing.getRange(`D2:D${last}`).format.columnWidth = 19;
timing.getRange(`E2:O${last}`).format.columnWidth = 13;
timing.freezePanes.freezeRows(13);
timing.freezePanes.freezeColumns(2);

styleSheet(frames, `A2:Q${records.length + 7}`);
frames.getRange('A2').values = [['Experiment 9 frame boundaries and cadence']];
frames.getRange('A2').format.font = { name: 'Arial', size: 16, bold: true, color: navy };
frames.getRange('A4').values = [['Frame zero is CSV start. Offsets are exclusive. External pauses are not included.']];
frames.getRange('A5').values = [['Source: parameters/stimulus_time_summary.csv. Positions use zero-based indices.']];
frames.getRange('A4:A5').format.font = { name: 'Arial', size: 10, italic: true, color: '#526273' };
frames.getRange('A7:Q7').values = [[
  'Stimulus', 'Hemifield', 'Frame rate (Hz)', 'Pre offset frame', 'Delay frames', 'Bout onset frame',
  'Bout offset frame', 'CSV frames', 'Flicker ON frames', 'Flicker OFF frames', 'Bout ON frames',
  'Bout OFF frames', 'Pre ON pulses', 'Pre positions', 'Bout positions', 'Dot identities', 'Trajectory file',
]];
header(frames, 'A7:Q7');
frames.getRange(`A8:Q${records.length + 7}`).values = records.map(r => [
  r.stimulus_key, r.hemifield, r.framerate, r.preconditioning_offset_frame, r.delay_frames,
  r.bout_onset_frame, r.bout_offset_frame, r.n_frames, r.flicker_on_frames, r.flicker_off_frames,
  r.bout_on_frames, r.bout_off_frames, r.preconditioning_pulses,
  JSON.stringify(r.preconditioning_positions), JSON.stringify(r.bout_positions), r.n_dots, r.trajectory_file,
]);
frames.getRange(`C8:M${records.length + 7}`).setNumberFormat('0');
frames.getRange(`A2:A${records.length + 7}`).format.columnWidth = 23;
frames.getRange(`B2:B${records.length + 7}`).format.columnWidth = 13;
frames.getRange(`C2:M${records.length + 7}`).format.columnWidth = 15;
frames.getRange(`N2:O${records.length + 7}`).format.columnWidth = 24;
frames.getRange(`P2:P${records.length + 7}`).format.columnWidth = 14;
frames.getRange(`Q2:Q${records.length + 7}`).format.columnWidth = 38;
for (let i = 0; i < records.length; i++) {
  if (i % 2) frames.getRange(`A${i + 8}:Q${i + 8}`).format.fill = '#F4F6F9';
}
frames.freezePanes.freezeRows(7);
frames.freezePanes.freezeColumns(2);
workbook.recalculate();

// Inspect the calculated totals against the generator's frame-derived records.
const actual = timing.getRange(`B${totalRow}:E${totalRow}`).values[0];
const expected = [records.length, records.reduce((s, r) => s + r.csv_duration_sec, 0),
  records.reduce((s, r) => s + r.pause_before_sec + r.pause_after_sec, 0),
  records.reduce((s, r) => s + r.duration_with_pauses_sec, 0)];
if (actual.some((value, i) => typeof value !== 'number' || Math.abs(value - expected[i]) > 1e-8)) {
  throw new Error('Excel totals do not agree with generated timing: ' + JSON.stringify({ actual, expected }));
}
const errors = await workbook.inspect({ kind: 'match', searchTerm: '#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!',
  options: { useRegex: true, maxResults: 20 }, maxChars: 2000 });
if (previewDir) {
  await fs.mkdir(previewDir, { recursive: true });
  for (const [sheetName, range] of [['Timing', `A1:O${last + 4}`], ['Frames', `A1:Q${records.length + 7}`]]) {
    const preview = await workbook.render({ sheetName, range, scale: 1.5, format: 'png' });
    await fs.writeFile(path.join(previewDir, sheetName + '.png'), new Uint8Array(await preview.arrayBuffer()));
  }
  const inspection = await workbook.inspect({ kind: 'table', range: 'Timing!A4:E7', include: 'values,formulas',
    tableMaxRows: 4, tableMaxCols: 5, maxChars: 2000 });
  await fs.writeFile(path.join(previewDir, 'verification.json'), JSON.stringify({ totals: actual, inspection, errors }, null, 2));
}
await fs.mkdir(path.dirname(outputPath), { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
// Keep the automatic export inspection sidecar out of the stimulus asset folder.
await fs.rm(outputPath + '.inspect.ndjson', { force: true });
console.log('Excel timing workbook saved: ' + outputPath);
