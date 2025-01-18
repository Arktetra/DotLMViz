<script lang="ts">
    
    const { min = 0, max = 1, step = 0.1, label = "", changeEventCb = null, inpStyle = "" } = $props()
    
    let inpVal: number = $state(step % 1 === 0 ? Math.floor(((max-min)/2)+min) : ((max-min)/2)+min);

    const updateInpVal = (v : number) => {
        if(changeEventCb) changeEventCb(v)
        inpVal = v;
    }
</script>


<div class="w-full flex flex-col mb-5">
    {#if label}
        <label class="text-theme-r text-sm font-bold font-main">{label}</label>
    {/if}
    <div class="w-full relative flex flex-row justify-between items-center text-gray-500">
        <span class="absolute text-ti font-bold -bottom-4 -start-1">{min}</span>
        <input 
            type="range"
            value={inpVal}
            {min} {max} {step}
            oninput={(e : any) => updateInpVal(e.target.value)}
            class="w-full accent-theme {inpStyle}"
        />
        <span class="absolute text-ti font-bold -bottom-4 -end-1">{max}</span>
        <span 
            style="left: {((inpVal - min)/(max - min))*100}%;"
            class={"absolute translate-x-[-50%] -bottom-4 text-ti-s text-center bg-theme text-theme-w rounded-md min-w-[1.8rem] "}
        >
            {inpVal}
        </span>
    </div>
</div>