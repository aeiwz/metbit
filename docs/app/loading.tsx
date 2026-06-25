export default function HomeLoading() {
  return (
    <div className="homePage homeSkeleton">
      <div className="homeThemeControl" />

      <section className="homeHero">
        <div className="homeHeroCopy">
          <div className="homeHeroMeta">
            <span className="skEl" style={{ width: 160, height: 16 }} />
            <span className="skEl" style={{ width: 200, height: 14 }} />
          </div>
          <span className="skEl" style={{ width: '70%', height: 44, marginTop: 8 }} />
          <span className="skEl" style={{ width: '90%', height: 16, marginTop: 4 }} />
          <span className="skEl" style={{ width: '80%', height: 16 }} />
          <span className="skEl" style={{ width: '60%', height: 16 }} />
          <div className="homeActions">
            <span className="skEl" style={{ width: 160, height: 42, borderRadius: 8 }} />
            <span className="skEl" style={{ width: 120, height: 42, borderRadius: 8 }} />
          </div>
        </div>
        <div className="homeHeroVisual">
          <div className="homeHeroLogo">
            <span className="skEl" style={{ width: 126, height: 34 }} />
          </div>
          <div className="homeHeroPanel">
            {[1,2,3].map(i => (
              <div key={i} className="homePanelRow">
                <span className="skEl" style={{ width: 100, height: 14 }} />
                <span className="skEl" style={{ width: 48, height: 14 }} />
              </div>
            ))}
          </div>
        </div>
      </section>

      <section className="homeHighlights">
        {[1,2,3,4].map(i => (
          <article key={i} className="homeCard">
            <span className="skEl" style={{ width: 32, height: 32, borderRadius: 8 }} />
            <span className="skEl" style={{ width: '60%', height: 18, marginTop: 4 }} />
            <span className="skEl" style={{ width: '90%', height: 14 }} />
            <span className="skEl" style={{ width: '75%', height: 14 }} />
          </article>
        ))}
      </section>

      <section className="homeOverview">
        <div className="homeOverviewCopy">
          <span className="skEl" style={{ width: 220, height: 28 }} />
          <div style={{ display: 'flex', flexDirection: 'column', gap: 10, marginTop: 16 }}>
            {[1,2,3,4].map(i => <span key={i} className="skEl" style={{ width: `${[90,80,95,70][i-1]}%`, height: 14 }} />)}
          </div>
        </div>
        <div className="homeOverviewPanel">
          {[1,2,3,4].map(i => (
            <div key={i} className="homePanelRow">
              <span className="skEl" style={{ width: 110, height: 14 }} />
              <span className="skEl" style={{ width: 60, height: 14 }} />
            </div>
          ))}
          <span className="skEl" style={{ width: '100%', height: 36, borderRadius: 8, marginTop: 8 }} />
        </div>
      </section>

      <section className="homeStats">
        {[1,2,3].map(i => (
          <div key={i} className="homeStatItem">
            <span className="skEl" style={{ width: 24, height: 24, borderRadius: 6 }} />
            <span className="skEl" style={{ width: 100, height: 36, marginTop: 4 }} />
            <span className="skEl" style={{ width: 140, height: 14 }} />
          </div>
        ))}
      </section>
    </div>
  )
}
