package matsimIntegration;

import org.matsim.core.controler.AbstractModule;

import com.google.inject.Scope;
import com.google.inject.name.Names;

import linktolinkBPR.LinkToLinks;


public class DNLDataCollectionModule extends AbstractModule{
	
	private LinkToLinks l2ls;
	private String fileLoc;
	
	public DNLDataCollectionModule(LinkToLinks l2ls,String fileLoc) {
		this.l2ls=l2ls;
		this.fileLoc=fileLoc;
	}
	
	public void install() {
		bind(LinkToLinks.class).toInstance(this.l2ls);
		bind(LinkToLinkTTRecorder.class).toInstance(new LinkToLinkTTRecorder(this.l2ls));
		bind(String.class).annotatedWith(Names.named("fileLoc")).toInstance(this.fileLoc);
		this.addEventHandlerBinding().to(LinkToLinkTTRecorder.class).asEagerSingleton();
		this.addControlerListenerBinding().to(DNLDataCollectionControlerListener.class);
		
	}

}
