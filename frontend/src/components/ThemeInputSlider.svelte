<script lang="ts">
    
    const { min = 0, max = 1, step = 0.1, label = "", changeEvent = null, style = "" } = $props()
    
    let inpVal: number = $state(step % 1 === 0 ? Math.floor(((max-min)/2)+min) : ((max-min)/2)+min);

    function updateInpVal(newVal : number)
    {
        if(changeEvent) changeEvent()
        inpVal = newVal;
    }
</script>


<div class="w-full flex flex-col mb-5">
    {#if label}
        <label class="text-theme text-[0.75rem] font-bold font-main">{label}</label>
    {/if}
    <div class="w-full relative flex flex-row justify-between items-center text-gray-500">
        <span class="absolute text-[0.75rem] font-bold -bottom-4 -start-1">{min}</span>
        <input 
            type="range"
            min={min}
            max={max}
            step={step}
            oninput={(e : any) => updateInpVal(e.target.value)}
            value={inpVal}
            class={"w-full accent-theme " + style}
        />
        <span class="absolute text-[0.75rem] font-bold -bottom-4 -end-1">{max}</span>
        <span 
            style="left: {((inpVal - min)/(max - min))*100}%;"
            class={"absolute translate-x-[-50%] -bottom-4 text-[0.70rem] text-center bg-theme text-theme-w rounded-md min-w-[1.8rem] "}
        >
            {inpVal}
        </span>
    </div>
</div>