import { Link } from 'react-router-dom'
import { Card, Pill, Stat } from '../components/ui'
import { IconArrow, IconGitHub } from '../components/Icons'
import { api } from '../lib/api'
import { useReference } from '../lib/hooks'
import { kg } from '../lib/format'

const REPO = 'https://github.com/wekesawgodwin/yield-prediction-and-agronomic-recomendation'

/**
 * Credits as the team states them. Started from the repository's commit
 * history, then corrected by the project lead — some of the work (the
 * presentation deck, sourcing the survey data) never lands as commits here.
 */
const CONTRIBUTORS = [
  {
    name: 'Wekesa Godwin',
    handle: 'wekesawgodwin',
    role: 'Weather pipeline, modelling, deployment & front end',
    detail:
      'Earth Engine rainfall and temperature features, the tuning and advanced-model notebooks, the district model, the API container and its Railway deployment, and the design of this front end.',
  },
  {
    name: 'Mohammed Ismail Abdi',
    handle: 'mohaski',
    role: 'Feature engineering & the backend API',
    detail: 'Feature engineering, then the prediction, model-summary and recommendation-layer endpoints this app calls — completing the backend API it runs on.',
  },
  {
    name: 'Ibrahim George',
    handle: 'ibrahimgeorge177-source',
    role: 'Data cleaning pipeline',
    detail: 'The cleaning pipeline that turns the raw survey export into the analysis table everything else is built on.',
  },
  {
    name: 'Mary G. Kahiga',
    handle: 'sonni-k',
    role: 'Data sourcing, understanding & quality',
    detail: 'Sourcing the survey data and the data-understanding work behind the feature set, then season-specific valid ranges for weed pressure, invalid-entry handling, and the data layout the notebooks read.',
  },
  {
    name: 'Trevor Amayi',
    handle: 'trevoramayi-debug',
    role: 'Documentation, presentation & integration',
    detail: 'Project documentation and the presentation deck, with Alvin Maina, and branch integration across the team.',
  },
  {
    name: 'Alvin Maina',
    handle: 'Zyrexn',
    role: 'Documentation & presentation deck',
    detail: 'Project documentation and the presentation deck, with Trevor Amayi.',
  },
]

const STEPS = [
  {
    n: '01',
    title: 'Survey becomes a clean table',
    body: '23,674 farm plots from the One Acre Fund MEL Agronomic Survey, 2016–2020, cleaned into one row per plot with validated ranges and documented rules.',
  },
  {
    n: '02',
    title: 'Weather is joined from satellites',
    body: 'CHIRPS rainfall and ERA5 temperature are pulled through Google Earth Engine for every site and season, then reduced to growing-season features and long-run climatology.',
  },
  {
    n: '03',
    title: 'Five models vote',
    body: 'LightGBM, XGBoost, Random Forest, Extra Trees and an entity-embedding neural network, each an average of its top configurations, blended with equal weights.',
  },
  {
    n: '04',
    title: 'Plots are averaged into districts',
    body: 'Individual plots are noisy; district means are not. Predictions are aggregated to a district-season mean with a calibrated interval.',
  },
]

export default function Home() {
  const { data: summary } = useReference('model-summary', api.modelSummary)
  const perf = summary?.expected_performance
  const seasons = summary?.validation || []
  const best = seasons.length ? seasons[seasons.length - 1] : null

  return (
    <div className="stack" style={{ gap: 0 }}>
      {/* --- hero --- */}
      <section>
        <span className="eyebrow">Kenya · maize · 2016–2020</span>
        <h1 style={{ marginTop: 10 }}>
          Forecasting district maize yield,
          <br />
          and what would actually raise it.
        </h1>
        <p className="lede" style={{ marginTop: 16 }}>
          A yield model built on the One Acre Fund MEL Agronomic Survey, joined to satellite
          rainfall and temperature. It predicts the <strong>mean yield of a district-season</strong>{' '}
          in kg/ha with an honest interval, and ranks the changes available to a single plot —
          planting date, fertiliser, seed choice — by the yield each is expected to add.
        </p>

        <div className="row" style={{ marginTop: 22, gap: 10 }}>
          <Link className="btn btn-primary" to="/forecast">
            Run a forecast <IconArrow width={16} height={16} />
          </Link>
          <Link className="btn" to="/advisor">Get recommendations</Link>
          <a className="btn btn-ghost" href={REPO} target="_blank" rel="noreferrer">
            <IconGitHub width={16} height={16} /> Source
          </a>
        </div>
      </section>

      {/* --- headline numbers --- */}
      <section className="section">
        <div className="grid grid-3">
          <Card>
            <Stat label="District R² (2020)" value={best ? best.r2?.toFixed(3) : '0.586'} tone="maize" />
            <p className="small" style={{ margin: '8px 0 0' }}>
              Out-of-time: fitted on earlier seasons, scored on one it had never seen.
            </p>
          </Card>
          <Card>
            <Stat label="Typical error" value={best ? kg(best.mae_kg_ph) : '329'} unit="kg/ha" tone="rain" />
            <p className="small" style={{ margin: '8px 0 0' }}>
              Mean absolute error on a district mean of roughly 3,100 kg/ha.
            </p>
          </Card>
          <Card>
            <Stat label="Plots behind it" value={kg(summary?.n_training_plots || 23674)} tone="leaf" />
            <p className="small" style={{ margin: '8px 0 0' }}>
              Across {summary?.train_seasons?.length || 5} seasons and{' '}
              {summary?.feature_defaults?.districts || 51} districts.
            </p>
          </Card>
        </div>
        {perf && (
          <p className="tiny" style={{ marginTop: 12 }}>
            Season-to-season the model lands between R² {perf.r2_range?.[0]} and {perf.r2_range?.[1]},
            with mean error {kg(perf.mae_range_kg_ph?.[0])}–{kg(perf.mae_range_kg_ph?.[1])} kg/ha. The
            season matters more than the algorithm.
          </p>
        )}
      </section>

      {/* --- how it works --- */}
      <section className="section">
        <h2>How it works</h2>
        <p className="small" style={{ marginTop: 6 }}>
          Four stages, each documented in a notebook in the repository.
        </p>
        <div className="grid grid-pairs" style={{ marginTop: 16 }}>
          {STEPS.map((s) => (
            <Card key={s.n}>
              <div className="row" style={{ gap: 10, alignItems: 'baseline' }}>
                <span className="mono" style={{ color: 'var(--maize)', fontSize: '0.85rem' }}>{s.n}</span>
                <h3>{s.title}</h3>
              </div>
              <p className="small" style={{ margin: '10px 0 0' }}>{s.body}</p>
            </Card>
          ))}
        </div>
      </section>

      {/* --- honesty section: this is the project's actual character --- */}
      <section className="section">
        <Card title="What this model will not do">
          <div className="grid grid-2">
            <div>
              <p className="small">
                <strong>It will not tell one farmer what their plot will yield.</strong> Roughly 82% of
                plot-to-plot variation lies <em>within</em> a district and season — soil, management
                detail, pest pressure and recall error the survey never measured. A perfect district oracle
                would still only score R² 0.18 at plot level.
              </p>
              <p className="small">
                Per-plot numbers are returned, but they are weak by construction and shown here with
                that warning attached.
              </p>
            </div>
            <div>
              <p className="small">
                <strong>Recommendations are associations, not experiments.</strong> Lever effects come
                from fixed-effects response curves with district and season absorbed — strong evidence
                of a relationship, not proof that changing the input causes the gain.
              </p>
              <p className="small">
                Everything is reported in kg/ha. The survey carries no price data, so nothing here is
                costed or budgeted.
              </p>
            </div>
          </div>
        </Card>
      </section>

      {/* --- contributors --- */}
      <section className="section">
        <h2>Acknowledgements</h2>
        <p className="small" style={{ marginTop: 6 }}>
          Built by six contributors, each credited with the part of the work they led.
        </p>
        <div className="grid grid-2" style={{ marginTop: 16 }}>
          {CONTRIBUTORS.map((c) => (
            <Card key={c.handle}>
              <div className="row" style={{ justifyContent: 'space-between', alignItems: 'flex-start' }}>
                <div>
                  <h3>{c.name}</h3>
                  <span className="eyebrow">{c.role}</span>
                </div>
                <a className="pill" href={`https://github.com/${c.handle}`} target="_blank" rel="noreferrer">
                  <IconGitHub width={12} height={12} /> {c.handle}
                </a>
              </div>
              <p className="small" style={{ margin: '12px 0 0' }}>{c.detail}</p>
            </Card>
          ))}
        </div>

        <Card style={{ marginTop: 16 }} title="Data & sources">
          <div className="chip-row" style={{ marginBottom: 12 }}>
            <Pill tone="pill-maize">One Acre Fund MEL Agronomic Survey 2021 release</Pill>
            <Pill tone="pill-rain">CHIRPS rainfall</Pill>
            <Pill tone="pill-rain">ERA5-Land temperature</Pill>
            <Pill>Google Earth Engine</Pill>
          </div>
          <p className="small" style={{ margin: 0 }}>
            Survey data is used for research and teaching purposes; the yield figures are farmer-reported
            and carry recall error, which is one reason the model is reported at district level. Weather
            features are derived from open satellite products via Earth Engine.
          </p>
        </Card>
      </section>
    </div>
  )
}
