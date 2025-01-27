type ScatterPlotPoint = {
    x: number,
    y: number,
    token: string
}

type HeatMapPoint = {
    x: number,
    y: number,
    source: string,
    destination: string,
    score: number
}

type ScatterPlotData = ScatterPlotPoint[]
type HeatMapData = HeatMapPoint[]